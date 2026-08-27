# AGENTS.md

## Project overview

JustRelax.jl is a Julia package providing matrix-free, pseudo-transient iterative solvers for geodynamic multiphysics problems. The main solver families are incompressible Stokes flow, variational/free-surface Stokes flow, DYREL dynamic relaxation, and thermal diffusion. The package supports 2D and 3D CPU execution, CUDA and AMDGPU backends, and MPI-based distributed runs.

Important companion packages are:

- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) for backend-agnostic kernels.
- [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl) for distributed grids and MPI.
- [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) for rheology and material properties.
- [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) for particle and phase-ratio methods.

## Repository map

- `src/`: package implementation and shared solver kernels.
- `src/common.jl`: common implementation included by the CPU and GPU 2D/3D variants. Shared functionality should generally be added here or in a file it includes.
- `src/stokes/`: standard Stokes solver implementation.
- `src/variational_stokes/`: variational Stokes and free-surface support.
- `src/DYREL/`: 2D dynamic-relaxation solver.
- `src/thermal_diffusion/`: pseudo-transient and explicit thermal solvers.
- `src/types/`, `src/grid/`, `src/boundaryconditions/`, and `src/IO/`: core data structures, grids, boundary conditions, and output/checkpointing.
- `ext/`: optional CUDA, AMDGPU, and Makie package extensions.
- `test/`: package test environment and test suite.
- `miniapps/`: standalone examples and benchmark applications with their own environment.
- `docs/`: Documenter/Vitepress documentation and documentation environment.

## Development rules

- Preserve the public API unless the change explicitly requires an API change. Deprecate before removing an established interface when practical.
- Keep kernels backend- and dimension-agnostic when possible. Do not include a 2D kernel file from a 3D module or vice versa.
- Follow the existing backend-trait dispatch pattern. CPU implementations normally live in shared code, while GPU extensions specialize dispatch and forward to the shared implementation.
- Be careful with array allocation: constructors and solver state must remain on the selected CPU/GPU backend.
- New kernels must use the project’s ParallelStencil conventions and respect the module’s `@init_parallel_stencil` initialization.
- Update relevant tests, miniapps, and documentation when changing a solver, public type, or user-facing behavior.
- Do not commit generated documentation builds, local manifests, benchmark outputs, or large simulation data unless the task explicitly calls for them.
- Preserve unrelated user changes in the working tree.

## Environments and commands

Run commands from the repository root. Use `--startup-file=no` for reproducible Julia checks.

### Load the package

```sh
julia --project=. --startup-file=no -e 'using JustRelax; println("loaded")'
```

### CPU tests

The test runner executes the regular tests in parallel and runs MPI tests separately with two ranks:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=test --startup-file=no test/runtests.jl
```

Run one test file directly when iterating on a focused change:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=test --startup-file=no test/test_diffusion2D.jl
```

The runner accepts test filters and `--backend=CUDA` or `--backend=AMDGPU` for accelerator testing. GPU tests require the corresponding hardware and Julia package/runtime setup:

```sh
JULIA_JUSTRELAX_BACKEND=CUDA julia --project=test --startup-file=no test/runtests.jl --backend=CUDA
JULIA_JUSTRELAX_BACKEND=AMDGPU julia --project=test --startup-file=no test/runtests.jl --backend=AMDGPU
```

### Formatting

Julia files are formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl), and CI checks formatting on pull requests:

```sh
git runic main              # show formatting differences
git runic --inplace .       # apply formatting
```

Format only relevant Julia files when possible, and review the resulting diff before committing.

### Documentation

Build the documentation with:

```sh
julia --project=docs --startup-file=no docs/make.jl
```

Documentation changes should link to existing guide pages and keep examples executable where practical.

### Miniapps

Instantiate or run examples using the `miniapps` environment, for example:

```sh
julia --project=miniapps --startup-file=no miniapps/subduction/2D/Subduction2D.jl
julia --project=miniapps --startup-file=no miniapps/convection/Particles2D/Convection2D.jl
```

Miniapps may require MPI, graphics, GPU, or additional system resources. Do not treat a resource-related failure as a package regression without checking the setup.

## Change verification

Before handing off a change:

1. Run the smallest relevant focused test while iterating.
2. Run the full CPU test suite for solver, type, dispatch, or kernel changes when feasible.
3. Run Runic on changed Julia files.
4. Build the docs for documentation or public API changes when feasible.
5. Review `git diff --check` and `git status`; do not include unrelated files.

For changes involving CUDA, AMDGPU, or MPI, report which backend/resource checks were actually run. CPU-only validation does not establish accelerator correctness.

## Pull requests

Use a descriptive PR title beginning with the appropriate tag, such as `[BUGFIX]`, `[ADDITION]`, or `[DOC]`. The PR should explain the motivation and include relevant tests, miniapp updates, API compatibility considerations, and documentation updates. Contributions follow the MIT license and the Developer Certificate of Origin described in `CONTRIBUTING.md`.
