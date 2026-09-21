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
- On GPU backends `runtests.jl` deletes the CPU-only tests (`test_variational_operators_2D`, `test_rheology`, `test_dyrel_solver_3D`, `test_dyrel_taylor_green_MPI`). A new test that cannot run on a device must be added there, or it fails GPU CI.
- `test_stokes_burstedde` and `test_VanKeken` are dropped from the default full run because they are slow; they run when named.
- Test-only packages (`ParallelTestRunner`, `Suppressor`, `SpecialFunctions`, `GeophysicalModelGenerator`, `Pkg`, `Test`) are declared in the root `Project.toml` (`[extras]` + `[targets]`); there is no `test/Project.toml`. Add a new one there, with a `[compat]` entry.

## Writing tests

- Copy the header of `test/test_diffusion2D.jl`: read `ENV["JULIA_JUSTRELAX_BACKEND"]`, load CUDA/AMDGPU accordingly, call `@init_parallel_stencil(<backend>, Float64, <dim>)`, and define the `backend` / `backend_JP` constants.
- Always add a test for new functionality. Prefer a focused unit test over a full solver run.
- Use the smallest grid that exercises the code, and derive sizes from `ni` / `size(arr)` instead of repeating literals.
- **3D tests that must pass on GPU use tiny grids.** ParallelStencil's block heuristic rounds up, so a `13×13×13` range launches `13×13×2 = 338` threads — over the 256 that the register-heavy fused DYREL stress kernel can launch with (`ERROR_LAUNCH_OUT_OF_RESOURCES`). A launched range of at most 256 cells is a single exact block and always fits (e.g. `6×5×4`, `5×4×3`); ranges with `x ≥ 32` and `y ≥ 8` give `32×8×1`. Do not use a `12³` grid in a 3D DYREL test.
- Build device arrays on the host and assign whole (`A .= PTArray(backend)(host)`). No element loops or `@allowscalar` on device arrays; move results to the host with `Array(…)` before asserting.
- Where a reference exists, test numerical accuracy against it: SolCx, SolKz, SolVi, Burstedde, Taylor–Green, Blankenbach, VanKeken, the Maxwell stress build-up.
- Silence solver output with `@suppress` (Suppressor) as neighbouring tests do.

## MPI tests

- An MPI test must also pass on one rank and when run standalone: `init_MPI = JustRelax.MPI.Initialized() ? false : true`.
- For a halo change copy `test_periodic_boundary_conditions_MPI.jl`: fill each rank's field with a rank-dependent constant, `update_halo!`, and assert that the boundary rows hold the *other* rank's value.
- GPU MPI coverage runs on the CSCS pipeline (`ci/cscs-gh200.yml`), which lists its MPI test files explicitly — add a new GPU-capable MPI test there too.

## Debugging

- A GPU-only failure: reproduce on CPU first. If CPU is green, suspect a name missing from a GPU module's import list ([backend-rules](backend-rules.md)) or a type-unstable/non-inlined call in a kernel ("dynamic invocation").
- Every test file needs a fresh process (`@init_parallel_stencil` runs once per module per session).
