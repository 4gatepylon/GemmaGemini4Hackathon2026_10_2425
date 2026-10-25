# lib

Swift sources compiled into every profiling tool: timestamped encoders (`RegionProfiler.swift`), noise recording (`Noise.swift`), spec parsing and expressions (`Spec.swift`, `Expr.swift`), statistics (`Stats.swift`), and the `kprof/1` output format (`KProf.swift`).

Any Metal program can compile these files in to time its own GPU work; see "Systems of kernels" in [../README.md](../README.md).
