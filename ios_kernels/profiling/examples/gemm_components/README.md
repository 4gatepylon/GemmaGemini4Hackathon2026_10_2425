# gemm_components

Ablation example: a simdgroup GEMM whose `SKIP_*` switches each remove one component (global loads, MMA, stores), so `kbench` variants measure each component's cost as the difference from the full kernel.
