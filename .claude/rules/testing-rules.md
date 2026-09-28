---
paths:
  - test/**/*.jl
---

# Testing Rules

How to run tests: the `running-tests` skill.

## How the suite is assembled (`test/runtests.jl`)

- Only `test_*.jl` files are collected, so a new test needs no registration — but a file not named `test_*.jl` is silently skipped (`deprecated_*.jl` are skipped on purpose).
- Files with `MPI` in the name are pulled out of the parallel run and executed one at a time under `mpiexec -n 2`. A test-name filter does not apply to them.
- Each non-MPI test runs in its own worker process, except the light tests named in `test_worker` (traits, types, conversions, mask, mini kernels, interpolations, boundary conditions, JustPIC interface), which run in the main process. A light test must not call `@init_parallel_stencil` or depend on IGG state; a new light test has to be added to that list.
- On GPU backends `runtests.jl` deletes the CPU-only tests (`test_variational_operators_2D`, `test_rheology`, `test_dyrel_solver_3D`, `test_dyrel_taylor_green_MPI`). These are legacy exclusions, not a pattern for new tests: all new or modified tests must run on CPU, CUDA, and AMDGPU. Fix backend incompatibilities rather than excluding tests.
- `test_stokes_burstedde` and `test_VanKeken` are dropped from the default full run because they are slow; they run when named.
- Test-only packages (`ParallelTestRunner`, `Suppressor`, `SpecialFunctions`, `GeophysicalModelGenerator`, `Pkg`, `Test`) are declared in the root `Project.toml` (`[extras]` + `[targets]`); there is no `test/Project.toml`. Add a new one there, with a `[compat]` entry.

## Writing tests

- Copy the header of `test/test_diffusion2D.jl`: read `ENV["JULIA_JUSTRELAX_BACKEND"]`, load CUDA/AMDGPU accordingly, call `@init_parallel_stencil(<backend>, Float64, <dim>)`, and define the `backend` / `backend_JP` constants.
- Always add a test for new functionality. Prefer a focused unit test over a full solver run.
- Use the smallest grid that exercises the code, and derive sizes from `ni` / `size(arr)` instead of repeating literals.
- Keep solver tests small, but retain grids that expose GPU launch rounding (e.g. 17³ vertex ranges). Register-heavy kernels may allow only 256 threads per block; use explicit bounded launch dimensions in the responsible launcher rather than shrinking tests to hide the bug.
- Scalar helper tests may use host reference values, but numerical code intended for kernels also needs a device-kernel test. Run affected tests on CUDA and AMDGPU when available and report unavailable hardware explicitly.
- Build device arrays on the host and assign whole (`A .= PTArray(backend)(host)`). No element loops or `@allowscalar` on device arrays; move ordinary device arrays to the host with `Array(…)` before asserting. For `CellArray`s use `CellArrays.CPUCellArray(…)` first. Never use `@allowscalar` to hide a failure.
- Where a reference exists, test numerical accuracy against it: SolCx, SolKz, SolVi, Burstedde, Taylor–Green, Blankenbach, VanKeken, the Maxwell stress build-up.
- Silence solver output with `@suppress` (Suppressor) as neighbouring tests do.

## MPI tests

- An MPI test must also pass on one rank and when run standalone: `init_MPI = JustRelax.MPI.Initialized() ? false : true`.
- For a halo change copy `test_periodic_boundary_conditions_MPI.jl`: fill each rank's field with a rank-dependent constant, `update_halo!`, and assert that the boundary rows hold the *other* rank's value.
- GPU MPI coverage runs on the CSCS pipeline (`ci/cscs-gh200.yml`), which lists its MPI test files explicitly — add a new GPU-capable MPI test there too.

## Debugging

- A GPU-only failure: reproduce on CPU first. If CPU is green, suspect a name missing from a GPU module's import list ([backend-rules](backend-rules.md)) or a type-unstable/non-inlined call in a kernel ("dynamic invocation").
- Every test file needs a fresh process (`@init_parallel_stencil` runs once per module per session).
