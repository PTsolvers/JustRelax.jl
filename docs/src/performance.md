```@raw html
---
aside: false
---
```

# Performance

Commit-by-commit benchmark history for JustRelax's pseudo-transient Stokes and thermal
diffusion solvers on the CPU and GPU backends. Main-branch results are recorded by Buildkite on
the `benchmark-data` branch; the harness, cases, and performance model are described in
[`benchmarking/README.md`](https://github.com/PTsolvers/JustRelax.jl/tree/main/benchmarking).

Runtime is measured. The effective memory throughput ``T_\mathrm{eff}`` uses the versioned
algorithmic model recorded with each result and is compared against a STREAM triad measured on
the same hardware.

```@raw html
<PerformanceDashboard />
```
