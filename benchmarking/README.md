# JustRelax performance benchmarks

This environment measures steady-state JustRelax solver performance independently from the
correctness test suite. Every timed case receives fresh solver state, runs one evaluation per
sample, and validates the result outside the timed region.

Run the CPU suite from the repository root:

```sh
julia --project=benchmarking benchmarking/setup.jl
julia --project=benchmarking --threads=4 benchmarking/run_benchmarks.jl
```

The setup step develops the repository checkout into the isolated benchmark environment.
Run it again after deleting or regenerating `benchmarking/Manifest.toml` or switching Julia
versions; otherwise Julia may resolve the registered JustRelax release instead of the checkout
being measured.

The runner writes `benchmark_results.json`. Each entry contains the fields required by
`github-action-benchmark` (`name`, `unit`, and `value`) together with timing dispersion,
allocations, throughput, problem parameters, package versions, hardware metadata, the git
revision, and whether the worktree was dirty.

## Cases

Each case runs a fixed number of pseudo-transient iterations (100 by default), independent
of convergence, so its cost does not change when a solver converges faster or slower. A change
in convergence rate is a correctness and robustness question for the test suite, not a
throughput one.

- `Stokes`: `solve!` for incompressible, linear viscous flow (infinite shear and bulk moduli)
  with a SolCx-like buoyancy field, a viscosity contrast of 10, and free-slip boundaries, on
  `128²` cells in 2D and `32³` in 3D. The residual is never evaluated.
- `Thermal diffusion`: `heatdiffusion_PT!` with unit properties, a hot bottom, a cold top, and
  insulated sides, on the same grids. The residual is evaluated once, after the last
  iteration.

Each result is validated: the solver must report the expected iteration count and leave
finite velocity, pressure, or temperature fields.

## Performance model

Wall time and allocations are measured. The memory traffic is the effective memory access
`A_eff` of Räss et al. (2022, *Geosci. Model Dev.* 15, 5757), a lower bound on the bytes one
iteration must move: each unknown field is read and written once, each known field is read
once, and every field is counted as `nᴰ` entries regardless of its staggering. Stresses,
fluxes, and iteration parameters are excluded.

FLOPs are counted per cell and iteration from the operations written in the solver kernels,
each kernel counted once per cell regardless of staggering. An FMA counts as two FLOPs; add,
subtract, multiply, divide, and reciprocal each count as one. Negation, integer arithmetic,
comparisons, indexing, grid-spacing reads, and control flow are excluded, as are boundary
conditions and halo updates. FLOP counts are independent of the element type.

| Benchmark | Unknowns | Knowns | `A_eff` per iteration | FLOPs per iteration |
| --- | --- | --- | ---: | ---: |
| Stokes (2D) | `Vx`, `Vy`, `P` | `η`, `ρg` | `8 nᴰ sizeof(T)` | `123 nᴰ` |
| Stokes (3D) | `Vx`, `Vy`, `Vz`, `P` | `η`, `ρg` | `10 nᴰ sizeof(T)` | `222 nᴰ` |
| Thermal diffusion (2D) | `T` | `T_old`, `K`, `ρCp` | `5 nᴰ sizeof(T)` | `38 nᴰ` |
| Thermal diffusion (3D) | `T` | `T_old`, `K`, `ρCp` | `5 nᴰ sizeof(T)` | `52 nᴰ` |

The Stokes counts cover the divergence (5 in 2D, 8 in 3D), the compressible pressure update
(24), the strain rate (13, 28), the visco-elastic stress update (47, 102), the velocity update
(32, 57), and the velocity-to-displacement copy (2, 3). With `nout` beyond `iterMax`, the
residual norms are never evaluated. The diffusion counts cover the damped heat-flux update
(22, 33) and the pseudo-transient temperature update (16, 19); the single residual evaluation
per solve (11, 14 per cell) is excluded.

Each JSON record reports `effective_bandwidth_gb_per_second`, the effective memory throughput
`T_eff = A_eff × iterations / t`, together with `arithmetic_intensity_flops_per_byte`,
`effective_flops_per_second`, and `effective_gflops_per_second`. The metadata records
`peak_memory_bandwidth_gb_per_second` and `peak_compute_gflops`, measured with a STREAM triad
and an eight-chain FMA kernel on the same device and element type; they are the roofline
ceilings. The CPU FMA kernel does not vectorize across items, so the CPU compute peak is a
lower bound. Measurements from hardware
counters must use a distinct `performance_metric_source` label rather than silently replacing
this model.

Useful options:

```sh
julia --project=benchmarking benchmarking/run_benchmarks.jl --samples=20
julia --project=benchmarking benchmarking/run_benchmarks.jl --group=Stokes
julia --project=benchmarking benchmarking/run_benchmarks.jl --output=out/results.json
julia --project=benchmarking benchmarking/run_benchmarks.jl --backend=CUDA
julia --project=benchmarking benchmarking/run_benchmarks.jl --precision=Float32
```

`--backend` accepts `CPU` (default), `CUDA`, or `AMDGPU`. GPU cases use the same problem sizes
as the CPU and record the device name in `metadata.device` and `metadata.hardware_fingerprint`.
`--precision` accepts `Float64` (default) or `Float32` and sets the element type of the case
fields and the STREAM probe. The element type is part of each benchmark name and is recorded
in `metadata.float_type`, so the two precisions form separate dashboard series.

## Comparing against a base revision

```sh
julia --project=benchmarking --threads=4 benchmarking/compare.jl --rev=main
julia --project=benchmarking --threads=4 benchmarking/compare.jl --rev=origin/main --group=Stokes
```

The script checks out `--rev` (default `main`) in a temporary git worktree, installs the
working tree's benchmark harness into it, and runs the suite on that revision and then on the
working tree, on the same machine and with the same arguments. It prints each benchmark's
median time, interquartile spread, candidate-to-baseline ratio, and allocations. Both runs
must share a backend, element type, and hardware fingerprint. A ratio within the printed
spread is not evidence of a change; repeat the comparison before acting on it.

Pull requests that touch `src/`, `ext/`, `benchmarking/`, or `Project.toml` run this
comparison on a shared `ubuntu-latest` runner (`.github/workflows/Benchmarks.yml`) and post
the table as a comment.

## Performance dashboard

The dashboard is the **Performance** page of the documentation (`docs/src/performance.md`,
rendered by `docs/src/components/PerformanceDashboard.vue`). The page reads
`benchmark_history.json` from the `benchmark-data` branch when it is viewed, so new results
appear without rebuilding the documentation. The history file is aggregated from per-commit
result files:

```sh
julia --project=benchmarking benchmarking/build_dashboard.jl \
    --input=out/benchmark_results.json \
    --output=out/benchmark_history.json
```

Repeat `--input=PATH` for historical result files. Inputs must identify unique commit,
backend, and hardware combinations; the builder fails instead of silently selecting between
duplicate runs.

## Continuous tracking

Buildkite runs the suite in `.buildkite/run_tests.yml` on the `cuda` and `rocm` queues (the
CUDA agent also runs the CPU backend) for commits on `main`, and uploads each
`benchmark_results_<label>.json` as a build artifact. Add `[skip benchmarks]` to a commit
message to skip them. When Buildkite reports the final status of a `main` commit,
`.github/workflows/PublishBenchmarks.yml`:

1. downloads that build's benchmark artifacts, which are public on JuliaGPU's Buildkite, and
   rejects any measured on a dirty worktree;
2. stores them as `results/<commit>/<label>.json` on the `benchmark-data` branch;
3. regenerates `benchmark_history.json` on that branch from every stored result.

Per-commit result files are the source of truth; `benchmark_history.json` and the page are
derived from them. The workflow uses only the repository's built-in `GITHUB_TOKEN`.

The publish workflow requires the `benchmark-data` branch; to create it:

```sh
git switch --orphan benchmark-data
echo '{"schema_version": 1, "repository_url": "https://github.com/PTsolvers/JustRelax.jl", "runs": []}' \
    > benchmark_history.json
git add -f benchmark_history.json
git commit -m "Initialize benchmark history"
git push origin benchmark-data
```

Shared CI agents are not dedicated benchmark machines, and GPU models can vary between
builds of the same queue. Compare results only within one hardware fingerprint, and derive
any regression threshold from repeated measurements on that fingerprint rather than from an
arbitrary percentage.

MPI, DYREL, variational Stokes, particle-coupled, and nonlinear-rheology benchmarks are not
yet covered.
