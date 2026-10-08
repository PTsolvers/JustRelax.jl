---
name: new-miniapp
description: Set up, validate and place a new JustRelax.jl model or benchmark script (a miniapp), with or without a reference paper. Use when creating a new simulation setup or reproducing a published model.
---

# New miniapp

Set up a model in stages and check each stage before adding the next. Conventions for the finished script: [miniapps-rules](../../rules/miniapps-rules.md). Running existing ones: [`miniapps-and-benchmarks`](../miniapps-and-benchmarks/SKILL.md).

## Step 1: Understand the case

- **Reproducing a paper:** extract *all* parameters — domain, resolution, physical constants, rheology and its units, boundary and initial conditions, forcing, duration, outputs. Check parameter tables, figure captions and coordinate conventions.
- **New case:** ask for the science goal. Clarify geometry and dimension, physics (thermal, elastic, plastic, free surface), rheology, run length, and which quantities to diagnose.

## Step 2: Choose the solver and dimension

| Need | Entry point | Notes |
|---|---|---|
| Visco-elasto-plastic Stokes, no free surface | `solve!` | fixed PT damping from `PTStokesCoeffs` |
| Free surface as sticky air, no surface tracking | `solve_VariationalStokes!` | needs a `RockRatio` refreshed each step |
| Strong heterogeneity or non-linear rheology, hand-tuned damping fails | `solve_DYREL!` / `solve_VariationalDYREL!` | self-tuned; needs a `DYREL` state built before the time loop |
| Heat only / thermal evolution | `heatdiffusion_PT!` | `ThermalArrays` has a ghost ring |

Which dimensions and backends a solver exists for is defined by the `include` lines in `src/JustRelax_CPU.jl` and `src/ext/*/{2D,3D}.jl` (for instance `solve_VariationalDYREL!` is 2D only). Solver arguments differ between families — read the docstring, and see `.agents/solvers.md`.

## Step 3: Start from the closest miniapp

| Setup | Copy from |
|---|---|
| Stokes, single phase, analytic check | `benchmarks/stokes2D/solcx/`, `benchmarks/stokes2D/shear_band/ShearBand2D.jl` |
| Particles, phases, advection | `benchmarks/stokes2D/sinking_block/SinkingBlock2D.jl`, `convection/Particles2D/` |
| Free surface | `benchmarks/stokes2D/free_surface_stabilization/`, `benchmarks/stokes2D/StickyAirSubduction/VariationalSubduction2D.jl` |
| Thermal only | `benchmarks/thermal_diffusion/diffusion/diffusion2D.jl` |
| DYREL | `DYREL2D/shear_band/ShearBand2D_DYREL.jl`, `DYREL3D/shear_band/ShearBand3D_DYREL.jl` |
| 3D | `benchmarks/stokes3D/shear_band/ShearBand3D.jl` |
| MPI | `benchmarks/stokes2D/shear_band/ShearBand2D_MPI.jl` |

Put the new script beside its relatives in `miniapps/<family>/<case>/`.

## Step 4: Geometry — check immediately

- `ni = (nx, ny[, nz])` is the number of **cells**; `grid = Geometry(ni, li; origin)`; `(; xci, xvi) = grid`. Build it after `init_global_grid` when running under MPI.
- Print or plot the extents, `di`, and the vertical orientation (the last dimension points up). Compare with the paper's figures.

## Step 5: Phases, rheology, boundary conditions

- Phases and rheology come from GeoParams `SetMaterialParams` and JustPIC particles/`PhaseRatios`. **Plot the phase distribution** before running (`plot_field`, `plot_particles` with a Makie backend loaded, or a heatmap of the ratios).
- Boundary conditions: `VelocityBoundaryConditions` takes exactly one of `no_slip`, `free_slip`, `periodic` per face (a face that is `false` everywhere is left untouched by `flow_bcs!`); periodic faces come in pairs, and a periodic top is incompatible with `free_surface = true`. Apply with `flow_bcs!` / `thermal_bcs!`.

## Step 6: Initial conditions — verify

- `extrema` of `T`, viscosity, `ρg`, velocity look physical; plot the spatial distribution.
- `thermal.T` includes a ghost ring: the interior aligned with `xci` is `T[2:end-1, 2:end-1]`.
- With variational solvers `stokes.P` is not a physical field outside the rock. Mask it before plotting or comparing, and do not seed it with a lithostatic profile in the air.

## Step 7: Short test run

- CPU, tiny grid, a few time steps, few iterations. Check for `NaN` in `stokes.V`, `stokes.P`, `thermal.T`; check the reported solver residuals decrease; check the flow develops in the expected direction.
- Use small default `nx, ny, nt` so the script doubles as a smoke test.

## Step 8: Progressive validation

Run a short simulation at moderate resolution, visualize, and compare with the analytical solution or the paper's early-time figures: flow direction and magnitude, stress and strain-rate patterns, temperature evolution.

## Step 9: Production run and comparison

Full resolution and duration; make the diagnostics the science goal needs. When reproducing a paper, match its figure format, colormaps, axis ranges and snapshot times.

## Step 10: Finish

- Format with Runic; keep output directories out of git.
- A benchmark with a published reference is worth a small fast test in `test/` ([testing-rules](../../rules/testing-rules.md)).
- A tutorial-style script that should appear in the docs uses the Literate conventions in [miniapps-rules](../../rules/miniapps-rules.md) and is added to `docs/make.jl` and its `pages`.

## Common problems

- **NaN blow-up:** time step too large (`compute_dt`), extreme viscosity contrast (set `viscosity_cutoff`), unstable initial conditions, too few pseudo-transient iterations (`iterMax`), inconsistent units.
- **Nothing happens:** phase ratios not refreshed (`update_phase_ratios!`), all boundary conditions `false`, wrong sign of gravity in `ρg`, `RockRatio` not updated.
- **Wrong orientation:** the vertical axis is the last dimension and points up; gravity is negative.
- **GPU:** scalar indexing of device arrays (build on host, or use kernels), `Array(…)` before plotting, branching in kernels.
- **Slow convergence with plasticity or strong contrasts:** try DYREL, or raise `iterMax` before tuning damping by hand.
