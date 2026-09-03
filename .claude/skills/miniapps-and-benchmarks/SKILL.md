---
name: miniapps-and-benchmarks
description: How to run JustRelax.jl miniapps (benchmarks, convection, subduction) to validate solver/physics changes end-to-end. Use when verifying a physics change beyond unit tests.
---

# Miniapps and benchmarks

`miniapps/` contains full model setups. They share one environment: `miniapps/Project.toml`.

## Layout

- `miniapps/benchmarks/` — validation cases with known solutions:
  - `stokes2D/`, `stokes3D/` — Stokes solver benchmarks (e.g. solkz, solcx, burstedde-type analytic solutions, shear bands).
  - `thermal_diffusion/`, `thermal_stress/` — heat-equation and thermo-mechanical cases.
- `miniapps/convection/` — mantle convection setups: `GlobalConvection2D_WENO5.jl`, `Particles2D`, `Particles2D_nonDim`, `Particles3D`, `RisingBlob3D`, `WENO5`.
- `miniapps/subduction/` — `2D/` and `3D/` subduction models (use GeophysicalModelGenerator for setup).

## Running one

```bash
cd miniapps
julia --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'   # first time, to run against local JustRelax
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. benchmarks/stokes2D/<case>/<script>.jl
```

Notes:
- Scripts choose the backend the same way tests do (env var / editing the `backend_JR` constant at the top). On this machine only CPU works.
- Convection/subduction runs are long (thousands of pseudo-transient iterations per timestep). For a smoke test, reduce `nx, ny` and the number of timesteps (`nt`) at the top of the script rather than waiting for full runs.
- Output (VTK via WriteVTK, checkpoints via JLD2/HDF5, figures if Makie is loaded) goes to a `figs*`/output directory created by the script — don't commit those.

## Which miniapp validates what

- Stokes solver / rheology changes → `benchmarks/stokes2D` analytic cases + `test_shearband2D` (plasticity).
- Thermal solver changes → `benchmarks/thermal_diffusion`, and `test_Blankenbach` / `test_VanKeken` in `test/` (community benchmarks with published reference values).
- Advection / WENO changes → `convection/WENO5`, `test_WENO5`.
- Particle/phase-ratio coupling → `convection/Particles2D`, `test_VanKeken`.
