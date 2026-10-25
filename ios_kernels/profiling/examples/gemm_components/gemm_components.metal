// A simdgroup_matrix GEMM, Y[T,M] = X[T,K] · W[M,K]^T (fp16 in, fp32 out), with
// compile-time switches that each remove one component, so `kbench` variants can
// measure a component's cost as the difference from the full kernel:
//   SKIP_LOADS  global -> threadgroup tile loads happen only for the first K step;
//               later steps reuse the stale tiles (same barriers, same MMA work)
//   SKIP_MMA    no simdgroup loads or MMAs; each thread still reads its tiles from
//               threadgroup memory into a checksum, so the global loads stay live
//   SKIP_STORE  only one 8x8 tile per simdgroup is written instead of sixteen
//               (the other fifteen stay live behind a branch that is never taken)
// Everything a variant still computes reaches memory, so the compiler can't drop it.
// Tiles as in ../matmul/matmul.metal: 4 simdgroups compute a 64x64 tile, K in steps of 32.

#include <metal_stdlib>
using namespace metal;

#ifndef SKIP_LOADS
#define SKIP_LOADS 0
#endif
#ifndef SKIP_MMA
#define SKIP_MMA 0
#endif
#ifndef SKIP_STORE
#define SKIP_STORE 0
#endif

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

kernel void gemm_components(device const half* W [[buffer(0)]],
                            device const half* X [[buffer(1)]],
                            device float*      Y [[buffer(2)]],
                            constant uint&     T [[buffer(3)]],
                            constant uint&     M [[buffer(4)]],
                            constant uint&     K [[buffer(5)]],
                            uint2 tg   [[threadgroup_position_in_grid]],
                            uint  tid  [[thread_index_in_threadgroup]],
                            uint  sg   [[simdgroup_index_in_threadgroup]],
                            uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup half As[G_BM * G_LD];
    threadgroup half Bs[G_BN * G_LD];
    const uint n0 = tg.x * G_BN, m0 = tg.y * G_BM;
    const uint sr = (sg / 2) * 32, sc = (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float checksum = 0.0f;

    device const half* Xt = X + (ulong)m0 * K;
    device const half* Wt = W + (ulong)n0 * K;
    for (uint k0 = 0; k0 < K; k0 += G_BK) {
        if (!SKIP_LOADS || k0 == 0) {
            load_tile(As, Xt + k0, K, tid);
            load_tile(Bs, Wt + k0, K, tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if SKIP_MMA
        checksum += float(As[tid % (G_BM * G_LD)]) + float(Bs[(tid * 7) % (G_BN * G_LD)]);
#else
        for (uint kk = 0; kk < G_BK; kk += 8) {
            simdgroup_half8x8 a[4], b[4];
            for (uint i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sr + 8 * i) * G_LD + kk, G_LD);
            for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], Bs + (sc + 8 * j) * G_LD + kk, G_LD, ulong2(0, 0), true);
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

#if SKIP_MMA
    Y[((ulong)(m0 + sr) * M + n0 + sc + lane) % ((ulong)T * M)] = checksum;
#elif SKIP_STORE
    // T is never 0 at runtime, but the compiler can't know that, so every accumulator
    // (and the MMA work behind it) stays live while only one tile is actually written.
    // (Folding the accumulators with thread_elements() instead changed the MMA codegen
    // and made the kernel ~3x slower: ablation switches must not touch the measured path.)
    if (T == 0) {
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_store(acc[i][j], Y + (ulong)(m0 + sr + 8 * i) * M + n0 + sc + 8 * j, M);
    } else {
        simdgroup_store(acc[0][0], Y + (ulong)(m0 + sr) * M + n0 + sc, M);
    }
#else
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], Y + (ulong)(m0 + sr + 8 * i) * M + n0 + sc + 8 * j, M);
#endif
}
