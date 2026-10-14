---
name: miniapps-and-benchmarks
description: How to run JustRelax.jl miniapps (benchmarks, convection, subduction, DYREL) to validate solver or physics changes end-to-end, and which one validates what. Use when verifying a change beyond unit tests.
---

# Miniapps and benchmarks

`miniapps/` holds full model setups that share one environment, `miniapps/Project.toml`. `miniapps/README.md` lists every script. To write a new one see [`new-miniapp`](../new-miniapp/SKILL.md); conventions are in [miniapps-rules](../../rules/miniapps-rules.md).

## Layout

- `benchmarks/stokes2D/`, `benchmarks/stokes3D/` — validation cases with known or published solutions: SolCx, SolKz, SolVi, Burstedde, Taylor–Green, Blankenbach, VanKeken, shear bands, elastic build-up, sinking block, sticky-air subduction, free-surface stabilization, shear heating. `RunStokesBench2D.jl` and `RunStokesBench3D.jl` dispatch across the suite.
- `benchmarks/thermal_diffusion/`, `benchmarks/thermal_stress/` — heat-equation and thermo-mechanical cases.
- `convection/` — mantle convection: `Particles2D`, `Particles2D_nonDim`, `Particles3D`, `RisingBlob3D`, `Plume3D`, `WENO5`.
- `subduction/` — `2D/` and `3D/` subduction models (set up with GeophysicalModelGenerator).
- `DYREL2D/`, `DYREL3D/` — the same families run with the self-tuned DYREL solver (shear band, convection, free surface, subduction, volcano, sinking block, Taylor–Green, Kelvin–Helmholtz, plume, …).

## Running one

From the repository root:

```bash
julia --project=miniapps --startup-file=no -e 'using Pkg; Pkg.instantiate()'    # once
julia --project=miniapps --startup-file=no miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl
```

- **Local checkout.** `miniapps/Project.toml` has a `[sources]` entry for `JustRelax` with an absolute path from the author's machine, so on another machine the environment does not point at your checkout. Point it there with `julia --project=miniapps -e 'using Pkg; Pkg.develop(path=".")'`, and afterwards check `git diff miniapps/Project.toml` and revert it — that change must not be committed.
- **Backend.** Scripts select it with constants near the top (`backend`, `backend_JP`, in some scripts an `isCUDA` flag); read the header before running. Use a device only if it is functional (`CUDA.functional()`); otherwise the CPU run is what you can verify locally, and you should say so.
- **MPI variants** (`*_MPI.jl`): launch with an MPI launcher, e.g. `mpiexec -n 2 julia --project=miniapps --startup-file=no <script>`.
- **Cost.** Convection and subduction runs need thousands of pseudo-transient iterations per time step. For a smoke test, reduce the resolution (`nx`, `ny`) and the number of time steps (`nt`) instead of waiting for a full run.
- **Output** (figures via Makie, VTK via WriteVTK, checkpoints via JLD2/HDF5) goes to a directory the script creates (`figdir`, `figs*`). Do not commit it.

## Which miniapp validates what

| Change | Run |
|---|---|
| Stokes solver or rheology | `benchmarks/stokes2D` analytic cases (`solcx`, `solkz`, `solvi`), `shear_band/` for plasticity, `elastic_buildup/` for elasticity; plus `test_shearband2D` |
| 3D Stokes | `benchmarks/stokes3D` (`solvi`, `burstedde`, `taylor_green`, `shear_band`) |
| Variational / free surface | `benchmarks/stokes2D/free_surface_stabilization/` (Crameri, Rayleigh–Taylor, plume) plus `test_variational_free_surface` |
| DYREL | the matching `DYREL2D/` / `DYREL3D/` script plus `test_dyrel*` |
| Thermal solver | `benchmarks/thermal_diffusion`, `test_Blankenbach`, `test_VanKeken` (community benchmarks with published reference values) |
| Advection / WENO | `convection/WENO5`, `test_WENO5` |
| Particle / phase-ratio coupling | `convection/Particles2D`, `test_VanKeken` |
| Halo exchange, MPI | the `_MPI` variants of the shear-band and diffusion cases |

A convergence-affecting change needs a miniapp or benchmark run, not only unit tests. Report which script, at what resolution, on which backend and rank count.

## Documentation pages

Three miniapps are the source of documentation pages through Literate.jl (`docs/make.jl`); editing them changes the page. See [miniapps-rules](../../rules/miniapps-rules.md) and the `build-docs` skill.
