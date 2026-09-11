# Solvers

Companion to [`AGENTS.md`](../AGENTS.md). Covers the four solver families, what they
expect, and how their iteration loops are structured. Calling conventions are in
[`api.md`](api.md), field layout in [`grid.md`](grid.md), distributed behavior in
[`mpi.md`](mpi.md).

All solvers are matrix-free accelerated pseudo-transient (APT) iterations: the steady
equations are augmented with pseudo-time derivatives and relaxed to a fixed point. They
mutate their state container in place and return a named tuple of iteration counts and
residual histories.

## Availability

| Solver | 2D | 3D | GPU 2D | GPU 3D |
|--------|----|----|--------|--------|
| `solve!` (Stokes) | ✓ | ✓ | ✓ | ✓ |
| `solve_VariationalStokes!` | ✓ | ✓ | ✓ | — |
| `solve_DYREL!`, `solve_VariationalDYREL!` | ✓ | — | ✓ | — |
| `heatdiffusion_PT!` | ✓ | ✓ | ✓ | ✓ |

The gaps are structural: `src/ext/CUDA/3D.jl` and `src/ext/AMDGPU/3D.jl` include only
`stokes/Stokes3D.jl`, and the DYREL solver files are included only by the 2D modules.
Calling an unavailable combination raises a `MethodError` rather than silently falling back.

## `solve!` — standard Stokes

`src/stokes/Stokes2D.jl`, `src/stokes/Stokes3D.jl`.

```julia
solve!(stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, igg;
       kwargs = (; iterMax = 50e3, nout = 500, viscosity_cutoff = (-Inf, Inf)))
```

`kwargs` is a **required** keyword and must be a NamedTuple (see [`api.md`](api.md)).
Four argument forms dispatch on the sixth positional argument:

| Sixth argument | Problem |
|----------------|---------|
| `phase_ratios::JustPIC.PhaseRatios`, then `rheology`, `args` | multi-phase visco-elasto-plastic (the general form) |
| `rheology::MaterialParams`, then `args` | single-phase visco-elasto-plastic |
| `G`, `K` | linear visco-elastic, constant moduli |
| `K` | linear, bulk modulus only |

`grid` may be replaced by the spacing `di` alone (an `NTuple` or NamedTuple); this routes
through `JustRelax.legacy_uniform_grid`, which rebuilds a uniform `Geometry` — under MPI it
reconstructs the *global* domain from `nx_g()`/`ny_g()`/`nz_g()`.

Keywords: `iterMax = 50e3`, `iterMin = 1e2`, `nout = 500`, `viscosity_cutoff = (-Inf, Inf)`,
`viscosity_relaxation = 1e-2`, `λ_relaxation = 0.2`, `strain_increment = false`,
`free_surface = false`, `b_width = (4, 4, 0)`, `verbose = true`. Defaults differ slightly
between the four forms — read the method you are calling.

One iteration, in order:

1. `compute_∇V!` — divergence of the velocity field.
2. pressure update — compressibility-penalized PT step (`src/stokes/PressureKernels.jl`).
3. strain rates, then the visco-elastic stress update with the Maxwell model and the
   Powell–Hestenes plastic multiplier `λ` (`src/stokes/StressKernels.jl`,
   `src/rheology/StressUpdate.jl`).
4. viscosity update from `rheology` + `phase_ratios`, clamped by `viscosity_cutoff` and
   relaxed by `viscosity_relaxation`.
5. velocity update from the momentum residual, scaled by the local PT viscosity `ηdτ/ητ`
   (`src/stokes/VelocityKernels.jl`), then `flow_bcs!` and `update_halo!`.

`ητ` is the local-maxima-smoothed preconditioner built by `compute_maxloc!`; it is
halo-exchanged once before the loop and after each viscosity update.

Convergence is checked every `nout` iterations on

```
errs = (‖Rx‖ / √((nx_g()-2)*(ny_g()-1)), ‖Ry‖ / √((nx_g()-1)*(ny_g()-2)), ‖RP‖ / √(nx_g()*ny_g()))
err  = maximum(errs)
```

with `‖·‖` a global (`norm_mpi`) L2 norm. The loop stops when `err/err_it1 < ϵ_rel` or
`err < ϵ_abs`, both from `pt_stokes`, after at least `iterMin` iterations. Some methods
normalize instead by the pressure range and `grid.max_li`; the linear methods use a
`while iter < 2 || (...)` guard rather than `iterMin`.

## `solve_VariationalStokes!` — free surface via a rock-ratio mask

`src/variational_stokes/Stokes2D.jl`, `Stokes3D.jl`.

```julia
solve_VariationalStokes!(stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, ϕ,
                         rheology, args, dt, igg;
                         air_phase = 1, viscosity_cutoff = (1e18, 1e24))
```

Accepts plain keywords or a bundled `kwargs = (; ...)`.

`ϕ::RockRatio` carries volume fractions on centers, vertices, and faces. Center fractions
weight pressure and normal stress, vertex fractions weight shear stress, and face fractions
weight the momentum rows; a row whose fraction vanishes is **eliminated**, not solved with
air properties. That is what lets a sticky-air free surface work without tracking the
surface explicitly. Refresh the mask each step with
`update_rock_ratio!(ϕ, phase_ratios, air_phase)`.

Extra keywords beyond `solve!`: `air_phase::Integer = 0` (phase excluded from material
averages; `0` disables), `strain_increment = false` (accumulate strain increments, for
large deformation), `λ_relaxation = 0.2`, `free_surface = false`.

Pressure convention: a cell whose center fraction is invalid has its pressure degree of
freedom eliminated, and `compute_variational_P!` sets `P[i,j] = RP[i,j] = 0` there. So
`stokes.P` is *not* a physical field outside the rock — mask it before plotting, comparing,
or integrating, and do not seed it with a lithostatic profile in the air.

## `solve_DYREL!` / `solve_VariationalDYREL!` — self-tuned dynamic relaxation

`src/DYREL/`, 2D only.

```julia
dyrel = DYREL(backend, stokes, rheology, phase_ratios, di, dt)      # before the time loop
solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
             iterMax_PH = 1e3, total_iterMax = 50e3)
```

Note the argument order differs from `solve!`: `ρg` and `dyrel` come second and third, and
`grid` comes after `args`. The variational form inserts `ϕ::RockRatio` after
`phase_ratios`, and its `DYREL` state must be constructed with the same `ϕ`.

Instead of fixed PT damping, DYREL estimates the largest eigenvalue of the iteration
operator from Gershgorin bounds (`src/DYREL/Gershgorin.jl`, `Gershgorin_VS.jl`) and
recomputes per-node pseudo-time steps `dτVx/dτVy` and damping `βV`, `αV`, `cV` at each
outer Powell–Hestenes iteration. This removes CFL/damping hand-tuning for strongly
heterogeneous or non-linear rheologies.

The loop is two-level: an outer Powell–Hestenes loop on the pressure/plastic multiplier
(`iterMax_PH`, `λ_relaxation_PH`, `pressure_relaxation`) around an inner dynamic-relaxation
loop on velocity (`iterMax_DR`, `λ_relaxation_DR`, `rel_drop`), with `total_iterMax`
bounding the sum. `verbose_PH` and `verbose_DR` control the two progress streams
separately. `linear_viscosity = true` skips the non-linear viscosity update.

## `heatdiffusion_PT!` — pseudo-transient thermal diffusion

`src/thermal_diffusion/DiffusionPT_solver.jl`.

```julia
heatdiffusion_PT!(thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
                  kwargs = (; igg, phase = phase_ratios, iterMax = 10e3, nout = 1e2, verbose = true))
```

`kwargs` is required here too. Two argument forms: precomputed `K::AbstractArray`,
`ρCp::AbstractArray`, or `rheology` + `args` (a NamedTuple, typically `(; T, P)`) with
optional `phase`. Entries of `args` sized like the thermal cell centers are read directly;
larger entries are offset by one to skip their ghost nodes.

Solves `∂T/∂t = ∇·(K∇T)/ρCp + H`. Each iteration computes the damped fluxes `qT` scaled by
`θr_dτ`, then advances `T` with the flux divergence and the source term scaled by `dτ_ρ`,
applies `thermal_bcs!` and `update_halo!(thermal.T)`. Convergence is
`norm_mpi(thermal.ResT) / √(prod(global grid size)) < pt_thermal.ϵ`.

Passing `stokes` refreshes `thermal.adiabatic` before the loop; passing `phase` recomputes
the PT coefficients from the local phase ratios each iteration. Shear heating comes from
`compute_shear_heating!` (`src/thermal_diffusion/ShearHeating.jl`).

An explicit forward-Euler alternative lives in
`src/thermal_diffusion/DiffusionExplicit.jl` (`ThermalDiffusion1D/2D/3D.solve!`), using a
precomputed diffusivity `κ = K/ρCp` from `ThermalParameters`.

## Editing a solver

- Kernels go in `src/common.jl` or a file it includes; the 2D and 3D kernel files are
  strictly separated and must not cross-include.
- A change to the shared iteration structure hits all four module variants and both GPU
  extensions — check that each still includes the file you touched.
- Solvers assume `stokes`, `thermal`, `ρg`, and `phase_ratios` are all on the same backend.
  Mixing raises a `MethodError` at the trait dispatch, which is the intended behavior.
- Convergence-affecting changes need a miniapp or benchmark run, not just unit tests; see
  `miniapps/benchmarks/`.
