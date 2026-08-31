# Miniapps

Runnable example scripts exercising JustRelax.jl on concrete problems, grouped
by folder. None of these are (yet) built into the documentation — clone the
repo, `]activate --temp` (or use the project's own `Project.toml`), and run a
script directly with `julia --project script.jl` (add `-p n`/an MPI launcher
for the `_MPI` variants).

## `benchmarks/stokes2D/` — 2D Stokes flow benchmarks

- `Blankenbach2D/` — Blankenbach thermal convection benchmark (`Benchmark2D_sgd.jl`,
  `Benchmark2D_sgd_scaled.jl`, `Benchmark2D_WENO5.jl` with WENO5 advection; rheology
  in `Blankenbach_Rheology.jl`/`Blankenbach_Rheology_scaled.jl`).
- `RunStokesBench2D.jl` — entry point that dispatches across the 2D Stokes benchmark suite.
- `StickyAirSubduction/` — subduction with a sticky-air free surface (`Subduction2D.jl`,
  `VariationalSubduction2D.jl` with variational free-surface stabilization; rheology files alongside).
- `VanKeken/VanKeken.jl` — Van Keken et al. subduction benchmark.
- `Volcano2D/` — caldera/volcano collapse (`Caldera2D.jl`, `Caldera2D_DYREL.jl`; setup and rheology alongside).
- `elastic_buildup/` — viscoelastic stress build-up under constant strain rate,
  with and without phases (`Elastic_BuildUp.jl`, `Elastic_BuildUp_phases_incompressible.jl`).
- `free_surface_stabilization/` — free-surface stabilization schemes: Crameri et al.
  benchmark (`Crameri2D.jl`), Rayleigh-Taylor instability (`RayleighTaylor2D.jl` and a
  variational-Stokes variant), and a plume rising to a free surface (`PlumeFreeSurface_2D.jl`
  and a variational-Stokes variant).
- `shear_band/` — localized shear band formation: Drucker-Prager (`ShearBand2D.jl`),
  Drucker-Prager cap (`ShearBand2D_DPCap.jl`), power-law rheology (`ShearBand2D_PowerLaw.jl`),
  strain softening (`ShearBand2D_softening.jl`), MPI (`ShearBand2D_MPI.jl`), variational
  Stokes (`ShearBand2D_variational.jl`, `ShearBand2D_variational_MPI.jl`), plus refined-mesh,
  displacement-formulation, strain-increment, and comparison variants.
- `shear_heating/` — shear heating during deformation (`Shearheating2D.jl`,
  `Shearheating2D_Plitho.jl`; rheology in `Shearheating_rheology.jl`).
- `sinking_block/` — dense block sinking through a viscous medium (`SinkingBlock2D.jl`,
  and a WENO5-advection variant).
- `solcx/`, `solkz/`, `solvi/` — SolCx, SolKz, SolVi analytical Stokes solutions used
  as convergence benchmarks (each folder has the driver, the analytical `_solution.jl`,
  and a `viz*.jl` plotting script; `SolViEl.jl` adds elasticity).

## `benchmarks/stokes3D/` — 3D Stokes flow benchmarks

- `RunStokesBench3D.jl` — entry point that dispatches across the 3D Stokes benchmark suite.
- `StickyAirSubduction/Subduction3D.jl` — 3D sticky-air subduction (setup/rheology alongside).
- `burstedde/Burstedde.jl` — Burstedde manufactured-solution benchmark (with `vizBurstedde.jl`).
- `shear_band/` — 3D shear localization (`ShearBand3D.jl`, MPI variant, and `MultipleInclusions3D.jl`).
- `shear_heating/Shearheating3D.jl` — 3D shear heating (rheology in `Shearheating_rheology.jl`).
- `solvi/SolVi3D.jl` — 3D SolVi analytical benchmark (with `vizSolVi3D.jl`).
- `taylor_green/TaylorGreen.jl` — 3D Taylor-Green vortex benchmark (with `vizTaylorGreen.jl`).

## `benchmarks/thermal_diffusion/diffusion/` — pure heat diffusion

- `diffusion2D.jl` / `diffusion3D.jl` — basic 2D/3D diffusion.
- `diffusion2D_periodic.jl` / `diffusion3D_periodic.jl` — periodic boundary conditions.
- `diffusion2D_inner_BCs.jl` — internal (interior) boundary conditions.
- `diffusion2D_multiphase.jl` / `diffusion3D_multiphase.jl` — multiple material phases.
- `diffusion2D_MPI.jl` / `diffusion3D_MPI.jl` / `diffusion2D_multiphase_MPI.jl` / `diffusion3D_multiphase_MPI.jl` — MPI/`ImplicitGlobalGrid` variants.

Smallest, simplest scripts in the repository — the natural first thing to run.

## `benchmarks/thermal_stress/` — thermal stress in a magma chamber

- `Thermal_Stress_Magma_Chamber_nondim.jl` / `_nondim3D.jl` — 2D/3D non-dimensionalized
  thermal stress around a cooling magma chamber.
- `Thermal_Stress_Magma_Chamber_nondim_VS.jl` — variational-Stokes variant.

## `convection/` — thermal convection

- `GlobalConvection2D_WENO5.jl` — whole-domain 2D convection with WENO5 advection.
- `Particles2D/`, `Particles2D_nonDim/`, `Particles3D/` — layered-rheology convection
  advected with particles, in 2D, non-dimensionalized 2D, and 3D (`Layered_convection2D*.jl`
  / `Layered_convection3D.jl`; rheology in `Layered_rheology.jl`; `_refined.jl` variants use a refined mesh).
- `Plume3D/` — 3D mantle plume rising through layered rheology (`Plume3D.jl`, MPI variant,
  rheology in `Plume3D_rheology.jl`).
- `RisingBlob3D/Blob3D.jl` — buoyant blob rising in 3D.
- `WENO5/WENO_convection2D.jl` — 2D convection with WENO5 advection (rheology in `Layered_rheology.jl`).

## `subduction/` — subduction zone models

- `2D/` — 2D subduction (`Subduction2D.jl`; MPI, variational-Stokes, non-dimensionalized,
  and restart variants; setup in `Subduction2D_setup.jl`/`_setup_MPI.jl`, rheology in
  `Subduction2D_rheology.jl`/`_rheology_ND.jl`).
- `3D/` — 3D subduction (`Subduction3D.jl`, MPI variant; setup and rheology alongside).

## `DYREL2D/` — self-tuned Accelerated Pseudo-Transient (DYREL) solver

Counterparts of several benchmarks above, driven by the self-tuned APT solver
instead of manually-tuned pseudo-transient parameters (see the "Self-tuned APT
solver" docs page).

- `StickyAirSubduction/` — sticky-air subduction (`Subduction2D.jl` and DYREL/non-dimensionalized variants; setup/rheology alongside).
- `convection/` — layered convection with DYREL (`GlobalConvection2D_DYREL_refined.jl`,
  `Layered_convection2D_DYREL.jl`, `Layered_convection2D_DYREL_refined.jl`; rheology in
  `Layered_rheology.jl`/`GlobalConvectionrheology .jl`).
- `shear_band/` — shear localization with DYREL (`ShearBand2D_DYREL.jl`, MPI and
  simple-shear-periodic variants, power-law variant).
- `shear_heating/Shearheating2D_DYREL.jl` — shear heating with DYREL (rheology in `Shearheating_rheology.jl`).
- `sinking_block/SinkingBlock2D.jl` — sinking block with DYREL.
- `subduction/` — subduction with DYREL (`Subduction2D_DYREL.jl`; setup/rheology alongside).
- `thermal_stress/Thermal_Stress_Magma_Chamber_nondim.jl` — magma chamber thermal stress with DYREL.
- `volcano/Caldera2D.jl` — caldera collapse with DYREL (setup/rheology alongside).
