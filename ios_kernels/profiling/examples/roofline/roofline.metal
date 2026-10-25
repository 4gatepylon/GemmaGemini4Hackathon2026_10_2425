// Roofline microbenchmarks: measure this GPU's practical peaks, so other
// results can be reported as "% of measured roof" instead of spec-sheet numbers.
//   read_bw / copy_bw / write_bw: streaming memory bandwidth (float4, grid-stride)
//   mma_peak: simdgroup_matrix FP16 x FP16 -> FP32 multiply-accumulate throughput
//   fma_peak: scalar FP32 FMA throughput
// Results are kept alive by stores the compiler can't prove dead.

#include <metal_stdlib>
using namespace metal;

#ifndef UNROLL
#define UNROLL 4        // independent loads in flight per thread per iteration
#endif
#ifndef MMA_ACC
#define MMA_ACC 8       // independent accumulators per simdgroup (hides MMA latency)
#endif

kernel void read_bw(device const float4* src  [[buffer(0)]],
                    device float*        sink [[buffer(1)]],
                    constant uint&       n4   [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint nth [[threads_per_grid]]) {
    float4 acc[UNROLL];
    for (uint u = 0; u < UNROLL; ++u) acc[u] = 0.0f;
    uint i = gid;
    for (; i + (UNROLL - 1) * nth < n4; i += UNROLL * nth)
        for (uint u = 0; u < UNROLL; ++u) acc[u] += src[i + u * nth];
    for (; i < n4; i += nth) acc[0] += src[i];
    float4 t = 0.0f;
    for (uint u = 0; u < UNROLL; ++u) t += acc[u];
    float s = t.x + t.y + t.z + t.w;
    if (s == -1.2345e30f) sink[gid & 1023] = s;  // never true for our inputs, keeps the loads
}

kernel void copy_bw(device const float4* src [[buffer(0)]],
                    device float4*       dst [[buffer(1)]],
                    constant uint&       n4  [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint nth [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += nth) dst[i] = src[i];
}

kernel void write_bw(device float4*  dst [[buffer(0)]],
                     constant uint&  n4  [[buffer(1)]],
                     uint gid [[thread_position_in_grid]],
                     uint nth [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += nth) dst[i] = float4(float(i));
}

// Each simdgroup runs `iters` rounds of MMA_ACC independent 8x8x8 MMAs
// (2*8*8*8 = 1024 FLOPs each). Operands come from memory so nothing folds.
kernel void mma_peak(device const half*  ab    [[buffer(0)]],   // 128 halfs: A then B
                     device float*       out   [[buffer(1)]],   // 64 * MMA_ACC floats per simdgroup
                     constant uint&      iters [[buffer(2)]],
                     uint tg  [[threadgroup_position_in_grid]],
                     uint sg  [[simdgroup_index_in_threadgroup]],
                     uint nsg [[simdgroups_per_threadgroup]]) {
    simdgroup_half8x8 a, b;
    simdgroup_load(a, ab, 8);
    simdgroup_load(b, ab + 64, 8);
    simdgroup_float8x8 c[MMA_ACC];
    for (uint j = 0; j < MMA_ACC; ++j) c[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint i = 0; i < iters; ++i)
        for (uint j = 0; j < MMA_ACC; ++j) simdgroup_multiply_accumulate(c[j], a, b, c[j]);
    device float* o = out + (ulong)(tg * nsg + sg) * 64 * MMA_ACC;
    for (uint j = 0; j < MMA_ACC; ++j) simdgroup_store(c[j], o + 64 * j, 8);
}

// 8 independent float4 FMA chains per thread: 2 * 4 * 8 = 64 FLOPs per iteration.
kernel void fma_peak(device const float* seed  [[buffer(0)]],
                     device float*       out   [[buffer(1)]],
                     constant uint&      iters [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    float4 m = float4(seed[0]), a = float4(seed[1]);
    float4 x[8];
    for (uint j = 0; j < 8; ++j) x[j] = float4(float(gid + j));
    for (uint i = 0; i < iters; ++i)
        for (uint j = 0; j < 8; ++j) x[j] = fma(x[j], m, a);
    float4 t = 0.0f;
    for (uint j = 0; j < 8; ++j) t += x[j];
    out[gid] = t.x + t.y + t.z + t.w;
}
