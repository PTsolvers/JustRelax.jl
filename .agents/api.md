# Public API

Companion to [`AGENTS.md`](../AGENTS.md). Covers what the package exports, how a model
script is assembled, and the dispatch rules a new entry point must follow. See
[`solvers.md`](solvers.md) for the solver loops, [`grid.md`](grid.md) for grid objects and
field layout, and [`mpi.md`](mpi.md) for distributed runs.

## Loading the package

A model loads the root module plus exactly one dimension-specific submodule:

```julia
using JustRelax, JustRelax.JustRelax2D   # or JustRelax.JustRelax3D
```

The submodule's `__init__` calls `@init_parallel_stencil(Threads, Float64, nDim)`. A script
or test that writes its own `@parallel` kernels must call `@init_parallel_stencil` itself,
with the same backend token and dimension, before those kernels are parsed.

Loading `CUDA` or `AMDGPU` **before** `JustRelax` activates the package extension
(`ext/JustRelaxCUDAExt.jl`, `ext/JustRelaxAMDGPUExt.jl`), which defines a second
`JustRelax2D`/`JustRelax3D` module carrying GPU methods. Entry-point names and signatures
are identical; the device is picked from the array type of the containers passed in.

`JustRelax` re-exports `ImplicitGlobalGrid` and `JustPIC`, so `init_global_grid`,
`update_halo!`, `nx_g`, `PhaseRatios`, and friends are in scope after `using JustRelax`.
`MPI` is reachable as `JustRelax.MPI`.

## Backends

| Tag | Array type | Requires |
|-----|------------|----------|
| `CPUBackend` | `Array` | — |
| `CUDABackend` | `CuArray` | `using CUDA` |
| `AMDGPUBackend` | `ROCArray` | `using AMDGPU` |

`PTArray(::Type{<:AbstractBackend})` maps a tag to its array type. Constructors take the
tag as first argument so allocation lands on the right device:

```julia
stokes    = StokesArrays(backend, ni)
thermal   = ThermalArrays(backend, ni)
pt_stokes = PTStokesCoeffs(li, di)
```

`ni` is the number of **cells**, as an `NTuple`.

## Trait dispatch

`backend(x)` returns a `BackendTrait` derived from the array type backing `x`, not from
`x`'s own type: `CPUBackendTrait`, `CUDABackendTrait`, `AMDGPUBackendTrait`, or
`NonCPUBackendTrait` for any other `AbstractArray`. `backend` is defined for `Array`,
`AbstractArray`, `Velocity`, `Displacement`, `Vorticity`, `SymmetricTensor`, `Residual`,
`Viscosity`, `ThermalArrays`, and `StokesArrays` (via `stokes.P`); anything else throws.

Every public function follows the same three-layer shape:

```julia
# 1. public entry point: detect the backend
solve!(stokes::JustRelax.StokesArrays, args...; kwargs) =
    solve!(backend(stokes), stokes, args...; kwargs)

# 2. CPU branch, in shared code
solve!(::CPUBackendTrait, stokes, args...; kwargs) = _solve!(stokes, args...; kwargs...)

# 3. GPU branch, in src/ext/CUDA/2D.jl — forwards to the same _solve!
JR2D.solve!(::CUDABackendTrait, stokes, args...; kwargs) = _solve!(stokes, args...; kwargs...)
```

Adding a public function means adding all three layers, plus the AMDGPU mirror. The
implementation itself belongs in `src/common.jl` or a file it includes, never in a
per-backend file — the extension modules `include` the same shared sources.

## Solver keyword convention

Two calling conventions coexist, and they are **not** interchangeable:

- `solve!` (Stokes) and `heatdiffusion_PT!` declare `; kwargs`, a *required* keyword. Options
  must be bundled: `solve!(...; kwargs = (; iterMax = 50e3, verbose = false))`.
- `solve_VariationalStokes!`, `solve_DYREL!`, and `solve_VariationalDYREL!` declare
  `; kwargs...` and normalize through `flatten_solver_kwargs` (`src/common.jl`), so plain
  keywords and a bundled `kwargs = (; ...)` both work, and may be mixed.

When adding a solver entry point, prefer the `kwargs...` + `flatten_solver_kwargs` form.

## Containers

`StokesArrays(backend, ni)` — `src/types/stokes.jl`, constructors in
`src/types/constructors/stokes.jl`. Fields: `P`, `P0`, `V`, `∇V`, `Q`, `τ`, `ε`, `ε_pl`,
`EII_pl`, `EVol_pl`, `ε_vol_pl`, `viscosity`, `τ_o`, `R`, `U`, `ω`, `Δε`, `∇U`, `λ`, `λv`,
`ΔPψ`. Component sizes are in [`grid.md`](grid.md).

`ThermalArrays(backend, ni)` — `src/types/heat_diffusion.jl`. `T`, `Told`, `ΔT` are
cell-centered **with one ghost node per boundary**, so `size(T) == ni .+ 2`;
`T[2:end-1, 2:end-1]` is the interior field aligned with `xci`. `qTx`/`qTy`/`qTz` sit on
faces, `H`, `shear_heating`, `adiabatic`, `dT_dt`, `ResT` on centers.

`PTStokesCoeffs(li, di; ϵ_rel, ϵ_abs, Re, CFL, r)` — pseudo-transient damping for Stokes.

`PTThermalCoeffs(...)` — pseudo-transient damping for diffusion; four constructors, taking
either precomputed `K`/`ρCp` arrays or `rheology` (with optional `phase_ratios`) plus `args`.

`RockRatio(backend, ni)` / `update_rock_ratio!(ϕ, phase_ratios, air_phase)` — the volume
fractions the variational solvers use to mask air cells.

`DYREL(backend, stokes, rheology, phase_ratios[, ϕ], di, dt; ...)` — dynamic-relaxation
state; must be built before the time loop.

`StressParticles`, `PrincipalStress`, `WENO5` — see `src/stress_rotation/`,
`src/stokes/PrincipalStresses.jl`, `src/types/weno.jl`.

## Boundary conditions

`VelocityBoundaryConditions`, `DisplacementBoundaryConditions`, and
`TemperatureBoundaryConditions` are keyword-only constructors taking one named tuple per
condition, with four faces in 2D (`left`, `right`, `top`, `bot`) and six in 3D (plus
`front`, `back`). For the flow conditions, `no_slip`/`free_slip`/`periodic` are mutually
exclusive per face; a face that is `false` everywhere is left untouched by `flow_bcs!`,
which is how a prescribed velocity field survives. Periodic faces must be enabled in pairs,
and a periodic top is incompatible with `free_surface = true`.

Apply with `flow_bcs!(stokes, bcs)` and `thermal_bcs!(thermal, bcs)`;
`pureshear_bc!` and `simpleshear_bc!` impose the corresponding background fields.

## Other exported functionality

- Rheology and phases: `compute_viscosity!`, `compute_viscosity_εII!`,
  `compute_viscosity_τII!`, `compute_ρg!`, `compute_melt_fraction!`,
  `compute_dissolved_volatiles!`, `fn_ratio`, `update_phase_ratios_2D!/3D!`,
  `compute_yieldfunction_phase`, `compute_plastic_gradients_phase`.
- Stress: `tensor_invariant!`, `accumulate_tensor!`, `accumulate_vol!`,
  `compute_principal_stresses[!]`, `rotate_stress!`, `rotate_stress_particles!`,
  `stress2grid!`.
- Interpolation: `vertex2center!`, `center2vertex!`, `velocity2vertex!`,
  `velocity2center!`, `shear2center!`, `velocity2displacement!`,
  `displacement2velocity!`.
- Advection and particles: `WENO_advection!`, `subgrid_characteristic_time!`,
  `update_phases_given_markerchain!`.
- Utilities: `compute_dt`, `compute_lithostatic_pressure!`, `multi_copy!`, `take`,
  `versioninfo`.
- Plotting hooks: `plot_particles`, `plot_field` — defined here, implemented in
  `ext/JustRelaxMakieExt.jl`; they need a Makie backend loaded.

## Field-access macros

Defined in `src/Utils.jl`, exported by `src/common.jl`:

```julia
@velocity(stokes)        # (Vx, Vy[, Vz])
@displacement(stokes)    # (Ux, Uy[, Uz])
@strain(stokes)          # strain-rate components
@stress(stokes)          # deviatoric stress components
@tensor_center(stokes.τ) # tensor components at cell centers
@residuals(stokes)       # (Rx, Ry[, Rz], ∇V)
@idx ni                  # index ranges for @parallel
@allocate, @copy, @add   # allocation and in-place helpers
```

`@index` (from `CellArraysIndexing`) is re-exported by the 2D/3D submodules because a bare
`@index` from `JustPIC` resolves to the KernelAbstractions index macro instead.

## I/O

`JustRelax.DataIO` loads automatically and exports `checkpointing_hdf5`,
`load_checkpoint_hdf5`, `checkpointing_jld2`, `load_checkpoint_jld2`, `save_hdf5`,
`save_data`, `metadata`, `center_coordinates`, `vertex_coordinates`, `VTKDataSeries`,
`save_vtk`, `save_pvtk`, `save_particles`, `save_marker_chain`. The parallel writers are
described in [`mpi.md`](mpi.md).

## Changing the API

- Do not break an exported name without a deprecation cycle; PRs are tagged
  `[BUGFIX]`/`[ADDITION]`/`[DOC]` and reviewed against `.github/PULL_REQUEST_TEMPLATE.md`.
- A new export must be added in `src/common.jl` (or the root module for cross-dimensional
  types), so all four module variants pick it up.
- `docs/make.jl` runs with `checkdocs = :exports`: an exported symbol without a docstring
  fails the docs build.
- Annotate arguments only as narrowly as the implementation requires — a `Matrix{Float64}`
  annotation silently locks out GPU arrays, views, and `Float32`.
