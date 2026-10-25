// Hand-written Metal kernels for the Gemma 4 E2B text decoder forward pass.
//
// Compiled at runtime by main.swift (device.makeLibrary(source:)); the host
// prepends `#define`s for the tunables below, so they can be swept without
// editing this file.
//
// Conventions:
//   - weights are fp16, row-major [out_features, in_features]; norm weights fp32
//   - decode activations / residual stream are fp32
//   - one KV head; each KV-owning layer has its own fp16 cache [max_ctx][head_dim]

#include <metal_stdlib>
using namespace metal;

#ifndef MV_NSG
#define MV_NSG 8        // simdgroups per threadgroup in matvec kernels
#endif
#ifndef MV_NR
#define MV_NR 4         // output rows per simdgroup in matvec kernels
#endif
#ifndef ATT_NSG
#define ATT_NSG 8       // simdgroups per threadgroup in attention
#endif

// gelu_pytorch_tanh. precise:: because fast tanh loses accuracy for large |x|.
static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// ---------------------------------------------------------------------------
// Scaled embedding lookup for T rows: out[t, i] = E[tokens[t], i] * scale
// (tokens already offset to the first row's position). Grid: (D, T).
// ---------------------------------------------------------------------------
kernel void embed_rows(device const half* E      [[buffer(0)]],
                       device const int*  tokens [[buffer(1)]],
                       device float*      out    [[buffer(2)]],
                       constant uint&     D      [[buffer(3)]],
                       constant float&    scale  [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= D) return;
    out[(ulong)gid.y * D + gid.x] = float(E[(ulong)tokens[gid.y] * D + gid.x]) * scale;
}

// ---------------------------------------------------------------------------
// Threadgroup-wide reductions. All threads get the result.
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
// Sandwich-norm residual update. One threadgroup per row.
//   x[r, i] = (x[r, i] + src[r, i] * rsqrt(mean(src[r]^2) + eps) * w[i]) * post
// `post` is the layer_scalar after the last residual add of a layer, else 1.
// ---------------------------------------------------------------------------
kernel void rmsnorm_add(device const float* src  [[buffer(0)]],
                        device const float* w    [[buffer(1)]],
                        device float*       x    [[buffer(2)]],
                        constant uint&      D    [[buffer(3)]],
                        constant float&     eps  [[buffer(4)]],
                        constant float&     post [[buffer(5)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint ntg  [[threads_per_threadgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    device const float* sr = src + (ulong)row * D;
    device float*       xr = x + (ulong)row * D;
    float ss = 0.0f;
    for (uint i = tid; i < D; i += ntg) ss += sr[i] * sr[i];
    ss = tg_sum(ss, scratch, sg, lane, ntg / 32);
    float scale = rsqrt(ss / float(D) + eps);
    for (uint i = tid; i < D; i += ntg) xr[i] = (xr[i] + sr[i] * scale * w[i]) * post;
}

// ---------------------------------------------------------------------------
// Per-layer-embedding (PLE) combine. One threadgroup (P threads) per (layer, row).
//   ple[t, l, :] = (rmsnorm(proj[t, l, :] * proj_scale) * w + ple[t, l, :]) / sqrt(2)
// `ple` holds the scaled token-identity embedding on entry.
// ---------------------------------------------------------------------------
kernel void ple_combine(device const float* proj       [[buffer(0)]],
                        device float*       ple        [[buffer(1)]],
                        device const float* w          [[buffer(2)]],
                        constant uint&      LP         [[buffer(3)]],
                        constant float&     proj_scale [[buffer(4)]],
                        constant float&     eps        [[buffer(5)]],
                        uint2 tg   [[threadgroup_position_in_grid]],
                        uint  tid  [[thread_index_in_threadgroup]],
                        uint2 tpg  [[threads_per_threadgroup]],
                        uint  sg   [[simdgroup_index_in_threadgroup]],
                        uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    const uint ntg = tpg.x;
    const ulong i = (ulong)tg.y * LP + tg.x * ntg + tid;
    float v = proj[i] * proj_scale;
    float ss = tg_sum(v * v, scratch, sg, lane, ntg / 32);
    ple[i] = (v * rsqrt(ss / float(ntg) + eps) * w[tid] + ple[i]) * M_SQRT1_2_F;
}

// ---------------------------------------------------------------------------
// Matrix-vector product, the decode workhorse (memory-bandwidth bound).
//   EPI 0: y[r] = W[r]·x     EPI 1: y[r] += W[r]·x     EPI 2: y[r] = gelu(W[r]·x) * aux[r]
// Each simdgroup owns MV_NR consecutive rows. Each lane streams 8 halfs
// (16 bytes) per row per iteration, so a simdgroup reads 512 contiguous bytes
// of each row per step. Requires K % 256 == 0 and M % (MV_NSG*MV_NR) == 0
// (checked on the host).
// ---------------------------------------------------------------------------
template <int EPI>
kernel void matvec_f16(device const half*  W   [[buffer(0)]],
                       device const float* x   [[buffer(1)]],
                       device float*       y   [[buffer(2)]],
                       constant uint&      M   [[buffer(3)]],
                       constant uint&      K   [[buffer(4)]],
                       device const float* aux [[buffer(5)]],
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
        if (lane == 0) {
            if (EPI == 0) y[r0 + r] = s;
            if (EPI == 1) y[r0 + r] += s;
            if (EPI == 2) y[r0 + r] = gelu_tanh(s) * aux[r0 + r];
        }
    }
}
template [[host_name("matvec_f16")]] kernel void matvec_f16<0>(
    device const half*, device const float*, device float*, constant uint&,
    constant uint&, device const float*, uint, uint, uint);
template [[host_name("matvec_f16_add")]] kernel void matvec_f16<1>(
    device const half*, device const float*, device float*, constant uint&,
    constant uint&, device const float*, uint, uint, uint);
template [[host_name("matvec_f16_gelu_mul")]] kernel void matvec_f16<2>(
    device const half*, device const float*, device float*, constant uint&,
    constant uint&, device const float*, uint, uint, uint);

// ---------------------------------------------------------------------------
// Fused GeGLU up-projection: h[r] = gelu_tanh(Wg[r]·x) * (Wu[r]·x)
// Same tiling as matvec_f16, but each simdgroup reads the matching rows of
// both matrices so the gate/up intermediates never touch memory.
// ---------------------------------------------------------------------------
kernel void matvec_gate_up_gelu(device const half*  Wg [[buffer(0)]],
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
        if (lane == 0) h[r0 + r] = gelu_tanh(g) * u;
    }
}

// ---------------------------------------------------------------------------
// Per-head QK/V-norm + RoPE (HF "rotate_half" convention) + KV-cache write.
// qkv rows are [q (NH*HD) | k (HD) | v (HD)] (KV-shared layers: q only), fp32.
// One threadgroup of HD/2 threads per (head, row); head index runs over the
// q heads, then k, then v. Thread i owns the rotation pair (i, i + HD/2).
//   q, k: x / rms(x) * w, then rotate pairs i < rope_pairs (the rest have
//         inv_freq 0 in HF, i.e. identity). q is written back in place.
//   v:    x / rms(x) (no weight).
// k and v go to the fp16 caches at position pos0 + row.
// ---------------------------------------------------------------------------
struct QKVParams {
    uint n_heads;
    uint head_dim;
    uint rope_pairs;
    uint row_len;    // floats per qkv row
    uint pos0;
    float eps;
};

kernel void qkv_norm_rope(device float*        qkv      [[buffer(0)]],
                          device half*         kcache   [[buffer(1)]],
                          device half*         vcache   [[buffer(2)]],
                          device const float*  q_w      [[buffer(3)]],
                          device const float*  k_w      [[buffer(4)]],
                          device const float*  inv_freq [[buffer(5)]],
                          constant QKVParams&  p        [[buffer(6)]],
                          uint2 tg   [[threadgroup_position_in_grid]],
                          uint  i    [[thread_index_in_threadgroup]],
                          uint2 tpg  [[threads_per_threadgroup]],
                          uint  sg   [[simdgroup_index_in_threadgroup]],
                          uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    const uint ntg = tpg.x;
    const uint h = tg.x, t = tg.y;
    const uint hd = p.head_dim, half_d = hd / 2;
    const uint pos = p.pos0 + t;
    device float* v = qkv + (ulong)t * p.row_len + h * hd;

    float a = v[i], b = v[i + half_d];
    float ss = tg_sum(a * a + b * b, scratch, sg, lane, ntg / 32);
    float scale = rsqrt(ss / float(hd) + p.eps);
    a *= scale;
    b *= scale;

    if (h > p.n_heads) {  // v
        vcache[(ulong)pos * hd + i] = half(a);
        vcache[(ulong)pos * hd + i + half_d] = half(b);
        return;
    }
    device const float* w = h < p.n_heads ? q_w : k_w;
    a *= w[i];
    b *= w[i + half_d];
    if (i < p.rope_pairs) {
        // precise:: because pos * inv_freq can be hundreds of radians.
        float ang = float(pos) * inv_freq[i];
        float c = precise::cos(ang), s = precise::sin(ang);
        float ra = a * c - b * s;
        float rb = b * c + a * s;
        a = ra;
        b = rb;
    }
    if (h < p.n_heads) {
        v[i] = a;
        v[i + half_d] = b;
    } else {
        kcache[(ulong)pos * hd + i] = half(a);
        kcache[(ulong)pos * hd + i + half_d] = half(b);
    }
}

// ---------------------------------------------------------------------------
// Causal (optionally sliding-window) attention over one KV head, one
// threadgroup per (query head, query row). Query row t (position pos0 + t)
// attends to cache positions [start, pos0 + t], start = max(0, pos - window + 1)
// for sliding layers. Used for decode (T = 1) and prefill (T = chunk rows).
//   1. scores[j] = scale * q·k_j   (one simdgroup per key, lanes split HD)
//   2. softmax over scores in threadgroup memory
//   3. out = sum_j p_j v_j         (simdgroups split keys; partial sums are
//                                   reduced through the same threadgroup memory)
// Threadgroup memory: max(n_keys, ATT_NSG * HD) floats.
// ---------------------------------------------------------------------------
struct AttnParams {
    uint n_heads;
    uint pos0;
    uint q_row_stride;    // floats between consecutive query rows
    uint out_row_stride;  // floats between consecutive output rows
    uint window;          // 0 = full causal
    float scale;
};

template <uint HD>
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
    constexpr uint EPL = HD / 32;  // elements per lane (8 or 16), contiguous
    threadgroup float scratch[32];

    const uint h = tgpos.x;
    const uint t = tgpos.y;
    const uint pos = p.pos0 + t;
    const uint start = (p.window && pos + 1 > p.window) ? pos + 1 - p.window : 0;
    const uint n = pos + 1 - start;
    const uint nthreads = ATT_NSG * 32;

    device const half* K = kcache + (ulong)start * HD + lane * EPL;
    device const half* V = vcache + (ulong)start * HD + lane * EPL;
    device const float* qh = q + (ulong)t * p.q_row_stride + h * HD + lane * EPL;
    float4 qv[EPL / 4];
    for (uint e = 0; e < EPL / 4; ++e) qv[e] = *(device const float4*)(qh + 4 * e) * p.scale;

    // 1. scores
    for (uint j = sg; j < n; j += ATT_NSG) {
        device const half4* kp = (device const half4*)(K + (ulong)j * HD);
        float s = 0.0f;
        for (uint e = 0; e < EPL / 4; ++e) s += dot(qv[e], float4(kp[e]));
        s = simd_sum(s);
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
    float4 acc[EPL / 4];
    for (uint e = 0; e < EPL / 4; ++e) acc[e] = 0.0f;
    for (uint j = sg; j < n; j += ATT_NSG) {
        device const half4* vp = (device const half4*)(V + (ulong)j * HD);
        float pj = scores[j];
        for (uint e = 0; e < EPL / 4; ++e) acc[e] += pj * float4(vp[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // everyone is done reading scores
    threadgroup float4* partial = (threadgroup float4*)(scores + sg * HD + lane * EPL);
    for (uint e = 0; e < EPL / 4; ++e) partial[e] = acc[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* o = out + (ulong)t * p.out_row_stride + h * HD;
    for (uint d = tid; d < HD; d += nthreads) {
        float tot = 0.0f;
        for (uint s2 = 0; s2 < ATT_NSG; ++s2) tot += scores[s2 * HD + d];
        o[d] = tot / sum;
    }
}
template [[host_name("attention_256")]] kernel void attention<256>(
    device const float*, device const half*, device const half*, device float*,
    constant AttnParams&, threadgroup float*, uint2, uint, uint, uint);
template [[host_name("attention_512")]] kernel void attention<512>(
    device const float*, device const half*, device const half*, device float*,
    constant AttnParams&, threadgroup float*, uint2, uint, uint, uint);

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

kernel void f32_to_f16(device const float* src [[buffer(0)]],
                       device half*        dst [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    dst[i] = half(src[i]);
}

// PLE gate activation for T rows: dst[t, i] = half(gelu(src[t, i]) * ple[t, layer, i]).
// Grid: (P, T). `ple` is bound at the layer's offset; LP = floats per ple row.
kernel void ple_gate_act(device const float* src [[buffer(0)]],
                         device const float* ple [[buffer(1)]],
                         device half*        dst [[buffer(2)]],
                         constant uint&      P   [[buffer(3)]],
                         constant uint&      LP  [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
    const ulong i = (ulong)gid.y * P + gid.x;
    dst[i] = half(gelu_tanh(src[i]) * ple[(ulong)gid.y * LP + gid.x]);
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

// Fused GeGLU GEMM: H[T, M] = half( gelu_tanh(X·Wg^T) * (X·Wu^T) ). Same tiling as
// gemm_f16 with a 64x32 output tile per threadgroup (two accumulator sets).
kernel void gemm_gate_up_gelu(device const half* Wg [[buffer(0)]],
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
    // matrices is opaque, so elementwise gelu(g)*u is done on a staged tile).
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
        H[(ulong)(m0 + r) * M + n0 + c] = half(gelu_tanh(Cg[e]) * Cu[e]);
    }
}
