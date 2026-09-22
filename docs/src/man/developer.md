# Developer guide

This page maps the implementation so contributors can find the right layer
before changing code. For local setup, tests, formatting, and pull-request
requirements, see [Contributing](@ref).

## Architecture at a glance

```text
JustRelax
├── src/JustRelax.jl        public types, backend tags, top-level exports
├── src/JustRelax_CPU.jl    CPU 2D and 3D modules
├── src/common.jl           code included by every backend/dimension module
├── src/stokes/             standard pseudo-transient Stokes solvers
├── src/variational_stokes/ variational and free-surface Stokes solvers
├── src/DYREL/              dynamic-relaxation solver
├── src/thermal_diffusion/  pseudo-transient and explicit thermal solvers
├── src/rheology/           material-property evaluation and derived fields
├── src/boundaryconditions/ boundary-condition types and kernels
├── src/grid/, src/types/   grids, state, traits, and constructors
└── ext/                    CUDA, AMDGPU, and Makie extensions
```

`JustRelax2D` and `JustRelax3D` are the modules users import after
`using JustRelax`. They include `common.jl`, then their dimension-specific
solver entry points. The shared file provides constructors, grid utilities,
material updates, boundary conditions, interpolation, and compute kernels.

## Backends and dimensions

JustRelax writes solver logic once using ParallelStencil and dispatches on a
backend trait. The CPU modules initialize ParallelStencil with `Threads`; the
CUDA and AMDGPU extensions re-include the same shared source after initializing
their own 2D or 3D module.

```text
array/state → backend(state) → CPU/CUDA/AMDGPU trait → shared implementation
```

The extensions own device-array allocation (`PTArray`) and trait dispatch.
Do not branch on `isa CuArray` or allocate plain `Array`s inside backend-generic
code. Read [Selecting the backend](@ref) for the user-facing setup, including
the separate JustRelax and JustPIC backend tags.

`@init_parallel_stencil` can run only once per module in a Julia session. A
change of backend or dimensionality therefore requires a fresh Julia process.
Never attempt to reinitialize ParallelStencil in an already-loaded solver
module.

## Adding a solver

1. Start from the closest existing family: standard Stokes in `src/stokes/`,
   variational/free-surface Stokes in `src/variational_stokes/`, DYREL in
   `src/DYREL/`, or thermal diffusion in `src/thermal_diffusion/`.
2. Keep 2D and 3D entry points separate. A 2D source file must not be included
   from a 3D module, or the reverse.
3. Put code that is genuinely independent of both dimension and device in
   `src/common.jl` or one of the files it includes. Reuse `src/MiniKernels.jl`
   for existing staggered-grid helpers.
4. Dispatch through the backend trait, following the public entry point → CPU
   trait → shared implementation pattern used by `solve!`. The extensions must
   include every source file required by the new solver.
5. Keep kernels GPU-safe: no allocations, dynamic dispatch, exceptions, or
   printing inside a kernel. Use `@parallel` for stencil kernels and
   `@parallel_indices` when explicit indices or cell-array access are needed.
6. Add focused CPU tests first. Run accelerator tests only on the corresponding
   hardware and report the backends actually checked.

## Adding material behavior

Most model rheology does **not** require a JustRelax change. Define phases and
their material properties with GeoParams' `SetMaterialParams` in the miniapp,
then pass the resulting `rheology` and, when needed, JustPIC `PhaseRatios` to
`compute_viscosity!`, `compute_ρg!`, and the solver.

Add code under `src/rheology/` only when JustRelax needs a new derived field or
solver-side evaluation. Follow the existing split:

- `Viscosity.jl` evaluates effective viscosity from the selected invariant and
  phase ratios.
- `BuoyancyForces.jl` evaluates density times gravity.
- `StressUpdate.jl`, `Melting.jl`, and `Solubility.jl` implement specialized
  updates.

Include a new file from `src/common.jl`, accept generic array and rheology
arguments unless the implementation requires more, and ensure the kernel works
for CPU, CUDA, and AMDGPU arrays. Add the smallest test that exercises the
new material path.

## Adding a boundary condition

First check whether the existing constructors cover the model:

- `TemperatureBoundaryConditions` supports no-flux, constant-flux,
  constant-value, periodic, and mask-based Dirichlet conditions.
- `VelocityBoundaryConditions` and `DisplacementBoundaryConditions` support
  no-slip, free-slip, periodic, and free-surface conditions.

For a new condition, add its type or kernel under `src/boundaryconditions/` and
include it through `BoundaryConditions.jl`, which is already shared by CPU and
GPU modules. Validate incompatible face combinations when constructing the
condition. Apply thermal conditions through `thermal_bcs!` and flow conditions
through `flow_bcs!`; preserve the trait-dispatch pattern so device-specific
modules can reuse the same implementation.

## Validation checklist

Before opening a pull request:

1. Run the smallest affected CPU test.
2. Format changed Julia files with Runic.
3. Build the documentation if public behavior or docs changed.
4. Review `git diff --check` and `git status` for unrelated files.

The [Contributing](@ref) page has the exact commands and backend-specific test
invocations.
