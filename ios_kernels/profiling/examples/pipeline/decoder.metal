// A Llama-style decode layer (one token) as a pipeline of kernels, for
// profiling a combination of kernels. Self-contained adaptation of the
// Llama-3.2-1B Metal kernel (kernels.metal in ../../../kernels/):
// rmsnorm -> qkv matvec -> rope + KV write -> attention -> wo matvec (+residual)
// -> rmsnorm -> fused gate/up SwiGLU -> down matvec (+residual).
// Activations fp32, weights and KV cache fp16.

#include <metal_stdlib>
using namespace metal;

#ifndef NSG
#define NSG 8        // matvec: simdgroups per threadgroup
#endif
#ifndef NR
#define NR 4         // matvec: rows per simdgroup
#endif
#ifndef ATT_NSG
#define ATT_NSG 8    // attention: simdgroups per threadgroup
#endif
#define HEAD_DIM 64

static inline float tg_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? scratch[lane] : 0.0f;
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

static inline float tg_max(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = lane < nsg ? scratch[lane] : -INFINITY;
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

kernel void rmsnorm(device const float* x   [[buffer(0)]],
                    device const float* w   [[buffer(1)]],
                    device float*       y   [[buffer(2)]],
                    constant uint&      D   [[buffer(3)]],
                    constant float&     eps [[buffer(4)]],
                    uint tid  [[thread_index_in_threadgroup]],
                    uint ntg  [[threads_per_threadgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    float ss = 0.0f;
    for (uint i = tid; i < D; i += ntg) ss += x[i] * x[i];
    ss = tg_sum(ss, scratch, sg, lane, ntg / 32);
    float scale = rsqrt(ss / float(D) + eps);
    for (uint i = tid; i < D; i += ntg) y[i] = x[i] * scale * w[i];
}

template <bool ADD>
kernel void matvec(device const half*  W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float*       y [[buffer(2)]],
                   constant uint&      M [[buffer(3)]],
                   constant uint&      K [[buffer(4)]],
                   uint tg   [[threadgroup_position_in_grid]],
                   uint sg   [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * NSG + sg) * NR;
    float acc[NR];
    for (uint r = 0; r < NR; ++r) acc[r] = 0.0f;
    for (uint k = lane * 8; k < K; k += 256) {
        float4 xa = *(device const float4*)(x + k);
        float4 xb = *(device const float4*)(x + k + 4);
        for (uint r = 0; r < NR; ++r) {
            device const half4* wp = (device const half4*)(W + (ulong)(r0 + r) * K + k);
            acc[r] += dot(float4(wp[0]), xa) + dot(float4(wp[1]), xb);
        }
    }
    for (uint r = 0; r < NR; ++r) {
        float s = simd_sum(acc[r]);
        if (lane == 0) y[r0 + r] = ADD ? y[r0 + r] + s : s;
    }
}
template [[host_name("matvec_f16")]] kernel void matvec<false>(
    device const half*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);
template [[host_name("matvec_f16_add")]] kernel void matvec<true>(
    device const half*, device const float*, device float*, constant uint&, constant uint&, uint, uint, uint);

kernel void gate_up_silu(device const half*  Wg [[buffer(0)]],
                         device const half*  Wu [[buffer(1)]],
                         device const float* x  [[buffer(2)]],
                         device float*       h  [[buffer(3)]],
                         constant uint&      M  [[buffer(4)]],
                         constant uint&      K  [[buffer(5)]],
                         uint tg   [[threadgroup_position_in_grid]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tg * NSG + sg) * NR;
    float ag[NR], au[NR];
    for (uint r = 0; r < NR; ++r) { ag[r] = 0.0f; au[r] = 0.0f; }
    for (uint k = lane * 8; k < K; k += 256) {
        float4 xa = *(device const float4*)(x + k);
        float4 xb = *(device const float4*)(x + k + 4);
        for (uint r = 0; r < NR; ++r) {
            ulong off = (ulong)(r0 + r) * K + k;
            device const half4* gp = (device const half4*)(Wg + off);
            device const half4* up = (device const half4*)(Wu + off);
            ag[r] += dot(float4(gp[0]), xa) + dot(float4(gp[1]), xb);
            au[r] += dot(float4(up[0]), xa) + dot(float4(up[1]), xb);
        }
    }
    for (uint r = 0; r < NR; ++r) {
        float g = simd_sum(ag[r]), u = simd_sum(au[r]);
        if (lane == 0) h[r0 + r] = g / (1.0f + exp(-g)) * u;
    }
}

struct RopeParams { uint n_heads, n_kv_heads, max_ctx, pos; };

// Grid (HEAD_DIM/2, n_heads + n_kv_heads): rotate q and k in place (rotate_half
// convention), write the rotated k and the v of position `pos` to the fp16 cache.
kernel void rope_kv(device float*        qkv    [[buffer(0)]],
                    device half*         kcache [[buffer(1)]],
                    device half*         vcache [[buffer(2)]],
                    constant RopeParams& p      [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, h = gid.y, hd2 = HEAD_DIM / 2;
    device float* v = qkv + h * HEAD_DIM;
    float ang = float(p.pos) * pow(500000.0f, -float(2 * i) / float(HEAD_DIM));
    float c = precise::cos(ang), s = precise::sin(ang);
    float a = v[i], b = v[i + hd2];
    float ra = a * c - b * s, rb = b * c + a * s;
    if (h < p.n_heads) {
        v[i] = ra;
        v[i + hd2] = rb;
    } else {
        uint kvh = h - p.n_heads;
        ulong base = ((ulong)kvh * p.max_ctx + p.pos) * HEAD_DIM;
        kcache[base + i] = half(ra);
        kcache[base + i + hd2] = half(rb);
        device const float* vv = qkv + (p.n_heads + p.n_kv_heads + kvh) * HEAD_DIM;
        vcache[base + i] = half(vv[i]);
        vcache[base + i + hd2] = half(vv[i + hd2]);
    }
}

struct AttnParams { uint n_heads, n_kv_heads, max_ctx, pos; float scale; };

// One threadgroup per query head: scores over keys [0, pos], softmax, weighted V.
kernel void attention(device const float* q      [[buffer(0)]],
                      device const half*  kcache [[buffer(1)]],
                      device const half*  vcache [[buffer(2)]],
                      device float*       out    [[buffer(3)]],
                      constant AttnParams& p     [[buffer(4)]],
                      threadgroup float*  scores [[threadgroup(0)]],
                      uint h    [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[32];
    threadgroup float2 partial[ATT_NSG][HEAD_DIM / 2];
    const uint n = p.pos + 1, kvh = h / (p.n_heads / p.n_kv_heads), nthreads = ATT_NSG * 32;
    device const half* K = kcache + (ulong)kvh * p.max_ctx * HEAD_DIM;
    device const half* V = vcache + (ulong)kvh * p.max_ctx * HEAD_DIM;
    float2 qv = float2(q[h * HEAD_DIM + 2 * lane], q[h * HEAD_DIM + 2 * lane + 1]) * p.scale;
    for (uint j = sg; j < n; j += ATT_NSG) {
        float2 kv = float2(*(device const half2*)(K + (ulong)j * HEAD_DIM + 2 * lane));
        float s = simd_sum(dot(qv, kv));
        if (lane == 0) scores[j] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint j = tid; j < n; j += nthreads) m = max(m, scores[j]);
    m = tg_max(m, scratch, sg, lane, ATT_NSG);
    float sum = 0.0f;
    for (uint j = tid; j < n; j += nthreads) { float e = exp(scores[j] - m); scores[j] = e; sum += e; }
    sum = tg_sum(sum, scratch, sg, lane, ATT_NSG);
    float2 acc = 0.0f;
    for (uint j = sg; j < n; j += ATT_NSG)
        acc += scores[j] * float2(*(device const half2*)(V + (ulong)j * HEAD_DIM + 2 * lane));
    partial[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float2 tot = 0.0f;
        for (uint s2 = 0; s2 < ATT_NSG; ++s2) tot += partial[s2][lane];
        tot /= sum;
        out[h * HEAD_DIM + 2 * lane] = tot.x;
        out[h * HEAD_DIM + 2 * lane + 1] = tot.y;
    }
}
