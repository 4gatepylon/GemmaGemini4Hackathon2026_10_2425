// Example kernels for single-kernel profiling. Self-contained versions of the
// shapes of work a decoder forward pass does:
//   matvec_f16:      y[M] = W[M,K] · x[K]   (decode; bandwidth bound)
//   gemm_naive_f16:  Y[T,M] = X[T,K] · W[M,K]^T, one thread per output (baseline)
//   gemm_simd_f16:   same with simdgroup_matrix 64x64 tiles (prefill; compute bound)
// W/X are fp16, accumulation and outputs fp32.

#include <metal_stdlib>
using namespace metal;

#ifndef NR
#define NR 4        // matvec: output rows per simdgroup
#endif
#ifndef NSG
#define NSG 8       // matvec: simdgroups per threadgroup
#endif

// Each lane streams 8 halfs (16 B) per row per step, so a simdgroup reads 512
// contiguous bytes of each of its NR rows per step. Needs K % 256 == 0 and
// M % (NR*NSG) == 0.
kernel void matvec_f16(device const half*  W [[buffer(0)]],
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
        if (lane == 0) y[r0 + r] = s;
    }
}

kernel void gemm_naive_f16(device const half* W [[buffer(0)]],
                           device const half* X [[buffer(1)]],
                           device float*      Y [[buffer(2)]],
                           constant uint&     T [[buffer(3)]],
                           constant uint&     M [[buffer(4)]],
                           constant uint&     K [[buffer(5)]],
                           uint2 gid [[thread_position_in_grid]]) {
    const uint m = gid.x, t = gid.y;
    if (m >= M || t >= T) return;
    device const half* w = W + (ulong)m * K;
    device const half* xr = X + (ulong)t * K;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) acc += float(w[k]) * float(xr[k]);
    Y[(ulong)t * M + m] = acc;
}

// 4 simdgroups (128 threads) compute a 64(T) x 64(M) tile, each a 32x32
// sub-tile of 4x4 8x8 accumulators. K is walked in steps of 32 through
// threadgroup memory. Needs T % 64 == 0, M % 64 == 0, K % 32 == 0.
#define G_BM 64
#define G_BN 64
#define G_BK 32
#define G_LD (G_BK + 8)

static inline void load_tile(threadgroup half* dst, device const half* src, uint K, uint tid) {
    const uint r = tid / 2, c = (tid % 2) * 16;
    device const half4* s = (device const half4*)(src + (ulong)r * K + c);
    threadgroup half4* d = (threadgroup half4*)(dst + r * G_LD + c);
    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}

kernel void gemm_simd_f16(device const half* W [[buffer(0)]],
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
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + (sc + 8 * j) * G_LD + kk, G_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Y + (ulong)(m0 + sr + 8 * i) * M + n0 + sc + 8 * j, M);
}
