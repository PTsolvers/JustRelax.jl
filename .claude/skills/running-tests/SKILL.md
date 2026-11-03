---
name: running-tests
description: Run JustRelax.jl tests — choose the tests a change is most likely to break, run single files, filtered or full suites, pick CPU/CUDA/AMDGPU, and run MPI tests. Use whenever running or debugging tests.
---

# Running tests

Run targeted tests one at a time, most-likely-to-break first, and fix each failure before moving on. The full suite is slow (solver tests run real pseudo-transient iterations): reserve it for solver, type, dispatch or kernel changes and let CI cover accelerators. Conventions for writing tests: [testing-rules](../../rules/testing-rules.md).

## Step 1: Pick the tests

Start from what changed. File names are descriptive, so treat the table as a starting point, and use `grep -rl "<changed_symbol>" test/` to find every test that touches a symbol.

| Changed area | Run first |
|---|---|
| `src/types/`, `src/JustRelax.jl` (traits, `PTArray`) | `test_types`, `test_traits`, `test_arrays_conversions` |
| `src/grid/`, `src/Utils.jl` | `test_grid2D`, `test_grid3D`, `test_Utils` |
| `src/MiniKernels.jl` | `test_mini_kernels` |
| `src/Interpolations.jl` | `test_Interpolations` |
| `src/boundaryconditions/` | `test_boundary_conditions2D`, `test_boundary_conditions3D`, `test_periodic_boundary_conditions_MPI` |
| `src/mask/`, `RockRatio` | `test_mask`, `test_rockratio` |
| `src/stokes/` | `test_stokes_solcx`, `test_stokes_solkz`, `test_shearband2D`, `test_stokes_taylor_green`, `test_stokes_elastic_buildup`, `test_sinking_block`; 3D: `test_stokes_solvi3D`, `test_shearband3D_MPI` (`test_stokes_burstedde` by name) |
| `src/rheology/` | `test_rheology`, `test_shearband2D`, `test_shearband2D_softening`, `test_shearband2D_DPCap` |
| `src/variational_stokes/` | `test_variational_free_surface`, `test_variational_operators_2D` |
| `src/DYREL/` | `test_dyrel`, `test_dyrel_kernels`, `test_dyrel_kernels_3D`, `test_dyrel_3D`, `test_dyrel_solver_3D`, `test_compute_local_stress_3d`, `test_dyrel_periodic_2D`, `test_dyrel_periodic_3D`, `test_variational_dyrel`, `test_shearband2D_DYREL`, `test_shearband2D_DPCap_DYREL` |
| `src/thermal_diffusion/` | `test_diffusion2D`, `test_diffusion3D`, `test_diffusion2D_multiphase`, `test_diffusion3D_multiphase`, `test_shearheating2D`, `test_shearheating3D`, `test_thermalstresses` |
| `src/advection/` | `test_WENO5` |
| `src/phases/`, `src/particles/`, JustPIC coupling | `test_phase_ratios3D`, `test_justpic_interface`, `test_VanKeken`, `test_Volcano2D` |
| `src/IO/` | `test_IO`, `test_IO_MPI` |
| Halo exchange, global reductions, lithostatic pressure | `test_periodic_boundary_conditions_MPI`, `test_lithostatic_pressure2D_MPI`, `test_lithostatic_pressure3D_MPI`, `test_diffusion2D_multiphase_MPI`, `test_shearband2D_MPI` |
| Convection / thermo-mechanical regressions | `test_Blankenbach`, `test_VanKeken` |
| `ext/`, `src/ext/` | the same tests on the device (see Backends) |

## Step 2: Run one file

`JULIA_JUSTRELAX_BACKEND` **must** be set (`CPU`, `CUDA` or `AMDGPU`). Test files read `ENV["JULIA_JUSTRELAX_BACKEND"]` at the top and throw a `KeyError` without it. Use one fresh Julia process per file, because each file calls `@init_parallel_stencil` in `Main`. In PowerShell, set `$env:JULIA_JUSTRELAX_BACKEND = "CPU"` first.

```bash
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no test/test_types.jl
```

- Test files do `push!(LOAD_PATH, "..")`, so they run against the local checkout.
- **There is no `test/Project.toml`** (the test environment was folded into the root `Project.toml`), so `--project=test` fails. `--project=.` works for every file that needs only the package's own dependencies (`Test` is a stdlib): e.g. `test_types`, `test_traits`, `test_arrays_conversions`, `test_mask`, `test_mini_kernels`, `test_rockratio`, `test_dyrel*`, `test_variational_*`, `test_boundary_conditions3D`.
- Many files `using Suppressor` (all the diffusion, shear-band, stokes, IO and grid tests; `test_WENO5` also `using SpecialFunctions`). Those packages exist only under `Pkg.test` (`[extras]`/`[targets]`). Check with `grep -l Suppressor test/<file>.jl`, then either use the filtered `Pkg.test` below or add them to a scratch environment stacked on the load path, which leaves both the project and your default environment untouched:

```bash
SCRATCH=$(mktemp -d)
julia --project=$SCRATCH -e 'using Pkg; Pkg.add(["Suppressor", "SpecialFunctions"])'
JULIA_LOAD_PATH="@:$SCRATCH:@stdlib" JULIA_JUSTRELAX_BACKEND=CPU \
  julia --project=. --startup-file=no test/test_diffusion2D.jl
```

The load-path separator is `:` on Linux/macOS and `;` on Windows. Overriding `JULIA_LOAD_PATH` drops the global environment, so re-add `@v#.#` if CUDA.jl or AMDGPU.jl (weakdeps, not project dependencies) live there.

## Step 3: Filtered or full suite

`Pkg.test` builds a temporary environment from `[targets]`, so it works from a clean checkout. This is what CI runs.

```bash
# a few tests, selected by name prefix
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test("JustRelax"; test_args=["test_diffusion2D"])'
# everything on the CPU
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
# accelerator
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test("JustRelax"; test_args=["--backend=CUDA"])'
```

How `test/runtests.jl` behaves:

- Positional arguments are name **prefixes** (`test_diffusion2D` also selects `test_diffusion2D_multiphase`); a leading `!` excludes.
- **The filter only applies to the parallel phase.** The `*MPI*` files are split out beforehand and *all* of them run afterwards, sequentially under `mpiexec -n 2`, whatever you filter. A filtered `Pkg.test` therefore still pays for the whole MPI phase; to run one MPI file, run it directly (Step 4).
- The default full run skips `test_stokes_burstedde` and `test_VanKeken` (slow). Name them to run them.
- `--backend=CUDA|AMDGPU` adds the GPU package and sets the environment variable itself. Four CPU-only tests are deleted on GPU (list in `runtests.jl`).
- Light tests (traits, types, conversions, mask, mini kernels, interpolations, boundary conditions, JustPIC interface) run in the main process; every other test gets its own worker.

## Step 4: MPI tests

```bash
mpiexec -n 2 julia --project=. --startup-file=no test/test_periodic_boundary_conditions_MPI.jl
```

Use MPI.jl's bundled launcher if no system MPI is configured. An MPI file that `using Suppressor` needs the scratch-environment recipe from Step 2. Each MPI test must also pass on one rank. CPU-only or 1-rank runs establish nothing about halo correctness ([mpi-rules](../../rules/mpi-rules.md)).

## Backends

- Check what is available at run time (`using CUDA; CUDA.functional()`, `using AMDGPU; AMDGPU.functional()`). Do not infer from the OS.
- With a working GPU: `JULIA_JUSTRELAX_BACKEND=CUDA julia --project=. --startup-file=no test/<file>.jl`, one process per file, with CUDA.jl visible on the load path.
- Without one, CPU-pass plus CI is the verification path: Buildkite CUDA (Julia 1.10 and 1) and AMDGPU jobs (`.buildkite/run_tests.yml`) and the CSCS GH200 CUDA + MPI pipeline (`ci/cscs-gh200.yml`). Say which backends and rank counts you actually ran.
- 3D tests that must pass on GPU use tiny grids ([testing-rules](../../rules/testing-rules.md)).

## Notes

- Quick smoke tests (seconds, no solver): `test_types`, `test_traits`, `test_arrays_conversions`, `test_mini_kernels`, `test_mask`. Physics regressions: `test_shearband2D`, `test_diffusion2D`, `test_Blankenbach`.
- If dependency resolution misbehaves after a Julia or package change, delete the local `Manifest.toml` (it is git-ignored) and re-instantiate.
- A GPU-only failure: reproduce on CPU first, then check the GPU import lists ([backend-rules](../../rules/backend-rules.md)).
