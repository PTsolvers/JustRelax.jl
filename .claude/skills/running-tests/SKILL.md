---
name: running-tests
description: How to run JustRelax.jl tests — single files, the full ParallelTestRunner suite, backend selection, and MPI tests. Use whenever running or debugging tests.
---

# Running tests

## Single test file (preferred during development)

Run one file directly against the local package:

```bash
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no test/test_shearband2D.jl
```

- `JULIA_JUSTRELAX_BACKEND` **must** be set (`CPU`, `CUDA`, or `AMDGPU`) — test files read `ENV["JULIA_JUSTRELAX_BACKEND"]` at the top and will throw a KeyError without it.
- Test files do `push!(LOAD_PATH, "..")`, and `test/runtests.jl` does `pushfirst!(LOAD_PATH, dirname(@__DIR__))`, so tests always run against the local checkout.
- Some test files need `Suppressor` and `ParallelTestRunner`, which live in the test environment. If a direct run fails on a missing package, use `julia --project=. -e 'using Pkg; Pkg.test(test_args=["test_shearband2D"])'` instead, or run with `--project=test` after instantiating it with the local JustRelax dev'd in.

## Full suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                            # CPU
julia --project=. -e 'using Pkg; Pkg.test(test_args=["--backend=CUDA"])' # GPU (adds CUDA via Pkg)
```

How `test/runtests.jl` works:
- Only files named `test_*.jl` are collected. `deprecated_*.jl` files are skipped.
- Non-MPI tests run in parallel via **ParallelTestRunner**; a worker is spawned per test except for a hardcoded list of light tests (traits, types, conversions, mask, mini kernels, interpolations, boundary conditions) that run without a worker.
- Files with `MPI` in the name (`test_*_MPI.jl`) are split out and run sequentially with `mpiexec -n 2`.
- You can filter: `Pkg.test(test_args=["test_diffusion2D"])` runs only matching tests (burstedde and VanKeken are excluded when filtering).

## Practical guidance

- The full suite is **slow** (solver tests run actual pseudo-transient iterations). Prefer one targeted test file. Quick smoke tests: `test_types`, `test_traits`, `test_arrays_conversions`, `test_boundary_conditions2D` (seconds, no solver). Physics regressions: `test_shearband2D`, `test_diffusion2D`, `test_Blankenbach`.
- MPI tests locally: `mpiexec -n 2 julia --project=. --startup-file=no test/test_diffusion2D_multiphase_MPI.jl` (use the MPI.jl-provided `mpiexecjl` if no system MPI is configured).
- GPU backends cannot be tested on this machine; CPU-pass + CI (CSCS GH200 pipeline in `ci/cscs-gh200.yml` runs CUDA + MPI tests) is the verification path.
