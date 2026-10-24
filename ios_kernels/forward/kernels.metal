// Hand-written Metal kernels for a Llama-style decoder forward pass.
//
// Compiled at runtime by main.swift (device.makeLibrary(source:)); the host
// prepends `#define`s for the tunables below, so they can be swept without
// editing this file.
//
// Conventions:
//   - weights are fp16, row-major [out_features, in_features]
//   - decode activations / residual stream are fp32
//   - KV cache is fp16, laid out [layer][kv_head][max_ctx][head_dim]

#include <metal_stdlib>
using namespace metal;

#ifndef MV_NSG
#define MV_NSG 8        // simdgroups per threadgroup in matvec kernels
#endif
#ifndef MV_NR
#define MV_NR 4         // output rows per simdgroup in matvec kernels
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 64     // attention kernels assume 2 elements per lane
#endif
#ifndef ATT_NSG
#define ATT_NSG 8       // simdgroups per threadgroup in attention
#endif

// ---------------------------------------------------------------------------
// Embedding lookup: x[i] = E[tokens[step], i]
// ---------------------------------------------------------------------------
kernel void embed(device const half*  E      [[buffer(0)]],
                  device const int*   tokens [[buffer(1)]],
                  device float*       x      [[buffer(2)]],
                  constant uint&      step   [[buffer(3)]],
                  constant uint&      D      [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= D) return;
    x[i] = float(E[(ulong)tokens[step] * D + i]);
}

// ---------------------------------------------------------------------------
// Threadgroup-wide sum. All threads get the result.
// ---------------------------------------------------------------------------
static inline float tg_sum(float v, threadgroup float* scratch,
                           uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? scratch[lane] : 0.0f;
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

static inline float tg_max(float v, threadgroup float* scratch,
                           uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? scratch[lane] : -INFINITY;
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

// ---------------------------------------------------------------------------
// RMSNorm over rows of length D. One threadgroup per row.
//   y[r, i] = x[r, i] * rsqrt(mean(x[r]^2) + eps) * w[i]
// OutT = float for decode, half for the prefill GEMM input.
// ---------------------------------------------------------------------------
template <typename OutT>
kernel void rmsnorm(device const float* x   [[buffer(0)]],
                    device const float* w   [[buffer(1)]],
                    device OutT*        y   [[buffer(2)]],
                    constant uint&      D   [[buffer(3)]],
                    constant float&     eps [[buffer(4)]],
                    uint row  [[threadgroup_position_in_grid]],
                    uint tid  [[thread_index_in_threadgroup]],
                    uint ntg  [[threads_per_threadgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * D;
    device OutT*        yr = y + (ulong)row * D;
    float ss = 0.0f;
    for (uint i = tid; i < D; i += ntg) ss += xr[i] * xr[i];
    ss = tg_sum(ss, scratch, sg, lane, ntg / 32);
    float scale = rsqrt(ss / float(D) + eps);
    for (uint i = tid; i < D; i += ntg) yr[i] = OutT(xr[i] * scale * w[i]);
}
template [[host_name("rmsnorm_f32")]] kernel void rmsnorm<float>(
    device const float*, device const float*, device float*, constant uint&,
    constant float&, uint, uint, uint, uint, uint);
template [[host_name("rmsnorm_f16")]] kernel void rmsnorm<half>(
    device const float*, device const float*, device half*, constant uint&,
    constant float&, uint, uint, uint, uint, uint);

// ---------------------------------------------------------------------------
// Matrix-vector product, the decode workhorse (memory-bandwidth bound).
//   y[r] (+)= sum_k W[r, k] * x[k]
// Each simdgroup owns MV_NR consecutive rows. Each lane streams 8 halfs
// (16 bytes) per row per iteration, so a simdgroup reads 512 contiguous bytes
// of each row per step. Requires K % 256 == 0 and M % (MV_NSG*MV_NR) == 0
// (checked on the host).
// ---------------------------------------------------------------------------
template <bool ADD>
kernel void matvec_f16(device const half*  W [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float*       y [[buffer(2)]],
                       constant uint&      M [[buffer(3)]],
                       constant uint&      K [[buffer(4)]],
                       uint tg   [[threadgroup_position_in_grid]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * MV_NSG + sg) * MV_NR;
    float acc[MV_NR];
    for (uint r = 0; r < MV_NR; ++r) acc[r] = 0.0f;

    for (uint k = lane * 8; k < K; k += 256) {
        float4 xa = *(device const float4*)(x + k);
        float4 xb = *(device const float4*)(x + k + 4);
        for (uint r = 0; r < MV_NR; ++r) {
            device const half4* wp = (device const half4*)(W + (ulong)(r0 + r) * K + k);
            acc[r] += dot(float4(wp[0]), xa) + dot(float4(wp[1]), xb);
        }
    }
    for (uint r = 0; r < MV_NR; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0) y[r0 + r] = ADD ? y[r0 + r] + s : s;
    }
}
template [[host_name("matvec_f16")]] kernel void matvec_f16<false>(
    device const half*, device const float*, device float*, constant uint&,
    constant uint&, uint, uint, uint);
template [[host_name("matvec_f16_add")]] kernel void matvec_f16<true>(
    device const half*, device const float*, device float*, constant uint&,
    constant uint&, uint, uint, uint);

// ---------------------------------------------------------------------------
// Fused SwiGLU up-projection: h[r] = silu(Wg[r]·x) * (Wu[r]·x)
// Same tiling as matvec_f16, but each simdgroup reads the matching rows of
// both matrices so the gate/up intermediates never touch memory.
// ---------------------------------------------------------------------------
kernel void matvec_gate_up_silu(device const half*  Wg [[buffer(0)]],
                                device const half*  Wu [[buffer(1)]],
                                device const float* x  [[buffer(2)]],
                                device float*       h  [[buffer(3)]],
                                constant uint&      M  [[buffer(4)]],
                                constant uint&      K  [[buffer(5)]],
                                uint tg   [[threadgroup_position_in_grid]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * MV_NSG + sg) * MV_NR;
    float ag[MV_NR], au[MV_NR];
    for (uint r = 0; r < MV_NR; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }

    for (uint k = lane * 8; k < K; k += 256) {
        float4 xa = *(device const float4*)(x + k);
        float4 xb = *(device const float4*)(x + k + 4);
        for (uint r = 0; r < MV_NR; ++r) {
            ulong off = (ulong)(r0 + r) * K + k;
            device const half4* gp = (device const half4*)(Wg + off);
            device const half4* up = (device const half4*)(Wu + off);
            ag[r] += dot(float4(gp[0]), xa) + dot(float4(gp[1]), xb);
            au[r] += dot(float4(up[0]), xa) + dot(float4(up[1]), xb);
        }
    }
    for (uint r = 0; r < MV_NR; ++r) {
        float g = simd_sum(ag[r]);
        float u = simd_sum(au[r]);
        if (lane == 0) h[r0 + r] = g / (1.0f + exp(-g)) * u;
    }
}

// ---------------------------------------------------------------------------
// RoPE (HF "rotate_half" convention) on Q and K, plus KV-cache write.
// qkv rows are [q (NH*HD) | k (NKV*HD) | v (NKV*HD)], fp32.
// Grid: (HD/2, NH + NKV, T). Row t is at absolute position pos0 + t.
// Q is rotated in place; K (rotated) and V are written to the cache as fp16.
// inv_freq already includes the Llama-3 frequency scaling (computed on host).
// ---------------------------------------------------------------------------
struct RopeParams {
    uint n_heads;
    uint n_kv_heads;
    uint max_ctx;
    uint pos0;
};

kernel void rope_kv(device float*        qkv      [[buffer(0)]],
                    device half*         kcache   [[buffer(1)]],
                    device half*         vcache   [[buffer(2)]],
                    device const float*  inv_freq [[buffer(3)]],
                    constant RopeParams& p        [[buffer(4)]],
                    uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.x;          // rotation pair index in [0, HD/2)
    const uint h = gid.y;          // head index over q heads then k heads
    const uint t = gid.z;          // row within this batch
    const uint half_d = HEAD_DIM / 2;
    const uint row_len = (p.n_heads + 2 * p.n_kv_heads) * HEAD_DIM;
    const uint pos = p.pos0 + t;

    device float* row = qkv + (ulong)t * row_len;
    device float* v = row + h * HEAD_DIM;   // q or k head being rotated

    // precise:: because pos * inv_freq can be hundreds of radians.
    float ang = float(pos) * inv_freq[i];
    float c = precise::cos(ang), s = precise::sin(ang);
    float a = v[i], b = v[i + half_d];
    float ra = a * c - b * s;
    float rb = b * c + a * s;

    if (h < p.n_heads) {
        v[i] = ra;
        v[i + half_d] = rb;
    } else {
        uint kvh = h - p.n_heads;
        ulong base = ((ulong)kvh * p.max_ctx + pos) * HEAD_DIM;
        kcache[base + i] = half(ra);
        kcache[base + i + half_d] = half(rb);
        device const float* vv = row + (p.n_heads + p.n_kv_heads + kvh) * HEAD_DIM;
        vcache[base + i] = half(vv[i]);
        vcache[base + i + half_d] = half(vv[i + half_d]);
    }
}

// ---------------------------------------------------------------------------
// Causal attention, one threadgroup per (query head, query row).
// Query row t attends to cache positions [0, pos0 + t]. Used for both decode
// (T = 1) and naive prefill (T = prompt length).
//   1. scores[j] = q·k_j / sqrt(HD)  (one simdgroup per key, lanes split HD)
//   2. softmax over scores in threadgroup memory
//   3. out = sum_j p_j v_j          (simdgroups split keys, reduce at the end)
// Threadgroup memory: scores[max_ctx] floats + ATT_NSG*HD floats.
// ---------------------------------------------------------------------------
struct AttnParams {
    uint n_heads;
    uint n_kv_heads;
    uint max_ctx;
    uint pos0;
    uint q_row_stride;    // floats between consecutive query rows
    uint out_row_stride;  // floats between consecutive output rows
    float scale;
};

kernel void attention(device const float* q        [[buffer(0)]],
                      device const half*  kcache   [[buffer(1)]],
                      device const half*  vcache   [[buffer(2)]],
                      device float*       out      [[buffer(3)]],
                      constant AttnParams& p       [[buffer(4)]],
                      threadgroup float*  scores   [[threadgroup(0)]],
                      uint2 tgpos [[threadgroup_position_in_grid]],
                      uint  tid   [[thread_index_in_threadgroup]],
                      uint  sg    [[simdgroup_index_in_threadgroup]],
                      uint  lane  [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    threadgroup float2 partial[ATT_NSG][HEAD_DIM / 2];

    const uint h = tgpos.x;
    const uint t = tgpos.y;
    const uint n = p.pos0 + t + 1;                    // keys visible to this query
    const uint kvh = h / (p.n_heads / p.n_kv_heads);  // GQA head mapping
    const uint nthreads = ATT_NSG * 32;

    device const half* K = kcache + (ulong)kvh * p.max_ctx * HEAD_DIM;
    device const half* V = vcache + (ulong)kvh * p.max_ctx * HEAD_DIM;
    device const float* qh = q + (ulong)t * p.q_row_stride + h * HEAD_DIM;
    float2 qv = float2(qh[2 * lane], qh[2 * lane + 1]) * p.scale;

    // 1. scores
    for (uint j = sg; j < n; j += ATT_NSG) {
        float2 kv = float2(*(device const half2*)(K + (ulong)j * HEAD_DIM + 2 * lane));
        float s = simd_sum(dot(qv, kv));
        if (lane == 0) scores[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. softmax
    float m = -INFINITY;
    for (uint j = tid; j < n; j += nthreads) m = max(m, scores[j]);
    m = tg_max(m, scratch, sg, lane, ATT_NSG);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += nthreads) {
        float e = exp(scores[j] - m);
        scores[j] = e;
        sum += e;
    }
    sum = tg_sum(sum, scratch, sg, lane, ATT_NSG);  // also acts as a barrier for scores

    // 3. weighted sum of V
    float2 acc = 0.0f;
    for (uint j = sg; j < n; j += ATT_NSG) {
        float2 vv = float2(*(device const half2*)(V + (ulong)j * HEAD_DIM + 2 * lane));
        acc += scores[j] * vv;
    }
    partial[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float2 tot = 0.0f;
        for (uint s2 = 0; s2 < ATT_NSG; ++s2) tot += partial[s2][lane];
        tot /= sum;
        device float* o = out + (ulong)t * p.out_row_stride + h * HEAD_DIM;
        o[2 * lane] = tot.x;
        o[2 * lane + 1] = tot.y;
    }
}

// ---------------------------------------------------------------------------
// Greedy sampling: tokens[out_idx] = argmax(logits). Single threadgroup.
// ---------------------------------------------------------------------------
kernel void argmax(device const float* logits  [[buffer(0)]],
                   device int*         tokens  [[buffer(1)]],
                   constant uint&      N       [[buffer(2)]],
                   constant uint&      out_idx [[buffer(3)]],
                   uint tid  [[thread_index_in_threadgroup]],
                   uint ntg  [[threads_per_threadgroup]],
                   uint sg   [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float best_v[32];
    threadgroup int   best_i[32];
    float bv = -INFINITY;
    int bi = 0;
    for (uint i = tid; i < N; i += ntg) {
        float v = logits[i];
        if (v > bv) { bv = v; bi = int(i); }
    }
    // simdgroup reduce (ties -> lowest index, matching torch.argmax)
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off);
        int   oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { best_v[sg] = bv; best_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = ntg / 32;
        bv = lane < nsg ? best_v[lane] : -INFINITY;
        bi = lane < nsg ? best_i[lane] : 0;
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            int   oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) tokens[out_idx] = bi;
    }
}

// ===========================================================================
// Prefill (batched) kernels
// ===========================================================================

// x[t, i] = E[tokens[t], i]   (tokens already offset to the chunk start)
kernel void embed_rows(device const half* E      [[buffer(0)]],
                       device const int*  tokens [[buffer(1)]],
                       device float*      x      [[buffer(2)]],
                       constant uint&     D      [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= D) return;
    x[(ulong)gid.y * D + gid.x] = float(E[(ulong)tokens[gid.y] * D + gid.x]);
}

kernel void f32_to_f16(device const float* src [[buffer(0)]],
                       device half*        dst [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    dst[i] = half(src[i]);
}

// ---------------------------------------------------------------------------
// Tiled GEMM on simdgroup matrices:  Y[T, M] (+)= X[T, K] · W[M, K]^T
//   X fp16 (activations, row-major), W fp16 (weights, row-major), Y fp32.
// Threadgroup = 4 simdgroups (128 threads) computing a 64(T) x 64(M) tile,
// each simdgroup a 32x32 sub-tile = 4x4 grid of 8x8 fp32 accumulators.
// K is walked in steps of 32 through threadgroup memory.
// Rows past T are computed on padding (host pads activation buffers to a
// multiple of 64 rows), so there are no bounds checks in the hot loop.
// ---------------------------------------------------------------------------
#define G_BM 64
#define G_BN 64
#define G_BK 32
#define G_LD (G_BK + 8)   // padded threadgroup row stride (halfs) to spread banks

// Cooperative load of a 64x32 fp16 tile (row stride K) into threadgroup memory.
static inline void load_tile(threadgroup half* dst, device const half* src, uint K, uint tid) {
    const uint r = tid / 2, c = (tid % 2) * 16;
    device const half4* s = (device const half4*)(src + (ulong)r * K + c);
    threadgroup half4* d = (threadgroup half4*)(dst + r * G_LD + c);
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

template <bool ADD>
kernel void gemm_f16(device const half* W [[buffer(0)]],
                     device const half* X [[buffer(1)]],
                     device float*      Y [[buffer(2)]],
                     constant uint&     T [[buffer(3)]],
                     constant uint&     M [[buffer(4)]],
                     constant uint&     K [[buffer(5)]],
                     uint2 tg  [[threadgroup_position_in_grid]],
                     uint  tid [[thread_index_in_threadgroup]],
                     uint  sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup half As[G_BM * G_LD];
    threadgroup half Bs[G_BN * G_LD];
    const uint n0 = tg.x * G_BN, m0 = tg.y * G_BM;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const half* Xt = X + (ulong)m0 * K;
    device const half* Wt = W + (ulong)n0 * K;
    for (uint k0 = 0; k0 < K; k0 += G_BK) {
        load_tile(As, Xt + k0, K, tid);
        load_tile(Bs, Wt + k0, K, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < G_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sr + 8 * i) * G_LD + kk, G_LD);
            // Bs holds W rows ([n][k]); load transposed to get the K x N operand.
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + (sc + 8 * j) * G_LD + kk, G_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) {
            device float* yp = Y + (ulong)(m0 + sr + 8 * i) * M + n0 + sc + 8 * j;
            if (ADD) {
                simdgroup_float8x8 old;
                simdgroup_load(old, yp, M);
                acc[i][j].thread_elements() += old.thread_elements();
            }
            simdgroup_store(acc[i][j], yp, M);
        }
}
template [[host_name("gemm_f16")]] kernel void gemm_f16<false>(
    device const half*, device const half*, device float*, constant uint&, constant uint&,
    constant uint&, uint2, uint, uint);
template [[host_name("gemm_f16_add")]] kernel void gemm_f16<true>(
    device const half*, device const half*, device float*, constant uint&, constant uint&,
    constant uint&, uint2, uint, uint);

// Fused SwiGLU GEMM: H[T, M] = half( silu(X·Wg^T) * (X·Wu^T) ). Same tiling as
// gemm_f16 with a 64x32 output tile per threadgroup (two accumulator sets).
kernel void gemm_gate_up_silu(device const half* Wg [[buffer(0)]],
                              device const half* Wu [[buffer(1)]],
                              device const half* X  [[buffer(2)]],
                              device half*       H  [[buffer(3)]],
                              constant uint&     T  [[buffer(4)]],
                              constant uint&     M  [[buffer(5)]],
                              constant uint&     K  [[buffer(6)]],
                              uint2 tg  [[threadgroup_position_in_grid]],
                              uint  tid [[thread_index_in_threadgroup]],
                              uint  sg  [[simdgroup_index_in_threadgroup]]) {
    threadgroup half As[G_BM * G_LD];
    threadgroup half Bs[G_BN * G_LD];   // rows 0..31 gate, 32..63 up
    const uint n0 = tg.x * (G_BN / 2), m0 = tg.y * G_BM;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 16;

    simdgroup_float8x8 ag[4][2], au[4][2];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) {
            ag[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            au[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }

    device const half* Xt = X + (ulong)m0 * K;
    for (uint k0 = 0; k0 < K; k0 += G_BK) {
        load_tile(As, Xt + k0, K, tid);
        // 32 gate rows then 32 up rows, using the same 128-thread loader layout.
        {
            const uint r = tid / 2, c = (tid % 2) * 16;
            device const half* src = (r < 32 ? Wg + (ulong)(n0 + r) * K : Wu + (ulong)(n0 + r - 32) * K) + k0 + c;
            device const half4* s = (device const half4*)src;
            threadgroup half4* d = (threadgroup half4*)(Bs + r * G_LD + c);
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < G_BK; kk += 8) {
            simdgroup_half8x8 a[4], bg[2], bu[2];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sr + 8 * i) * G_LD + kk, G_LD);
            for (uint j = 0; j < 2; ++j) {
                simdgroup_load(bg[j], Bs + (sc + 8 * j) * G_LD + kk, G_LD, ulong2(0, 0), true);
                simdgroup_load(bu[j], Bs + (32 + sc + 8 * j) * G_LD + kk, G_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 2; ++j) {
                    simdgroup_multiply_accumulate(ag[i][j], a[i], bg[j], ag[i][j]);
                    simdgroup_multiply_accumulate(au[i][j], a[i], bu[j], au[i][j]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue through threadgroup memory (the per-lane layout of simdgroup
    // matrices is opaque, so elementwise silu(g)*u is done on a staged tile).
    threadgroup float Cg[G_BM * (G_BN / 2)];
    threadgroup float Cu[G_BM * (G_BN / 2)];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 2; ++j) {
            simdgroup_store(ag[i][j], Cg + (sr + 8 * i) * (G_BN / 2) + sc + 8 * j, G_BN / 2);
            simdgroup_store(au[i][j], Cu + (sr + 8 * i) * (G_BN / 2) + sc + 8 * j, G_BN / 2);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < G_BM * (G_BN / 2); e += 128) {
        const uint r = e / (G_BN / 2), c = e % (G_BN / 2);
        float g = Cg[e], u = Cu[e];
        H[(ulong)(m0 + r) * M + n0 + c] = half(g / (1.0f + exp(-g)) * u);
    }
}
