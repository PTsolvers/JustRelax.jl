---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# MPI Rules

Distributed runs use ImplicitGlobalGrid (re-exported by JustRelax); every array a solver touches is the **local** subdomain plus overlap layers. Deeper reference: `.agents/mpi.md`.

- **Halos.** A field that a stencil reads across a rank boundary needs its own `update_halo!` at the same point in the loop. Order: apply boundary conditions first, then exchange the halo. A missing exchange only shows up under `mpiexec -n 2`, near the internal boundaries.
- **Global reductions.** Every convergence criterion, time-step bound and printed diagnostic goes through `norm_mpi`, `maximum_mpi`, `minimum_mpi`, `sum_mpi`, `mean_mpi` (in `src/Utils.jl`, not exported) — never a bare `norm`/`maximum` on a local array. Otherwise ranks disagree on when to stop and the run deadlocks or diverges silently.
- **Global sizes.** Normalize residuals by the global problem size (`nx_g()`, `ny_g()`, `nz_g()`), not the local array size.
- **Time step.** Use the `igg`-aware `compute_dt(stokes, di, dt_diff, igg)` in distributed runs; without `igg` ranks pick different time steps.
- **Grid order.** Build `Geometry` *after* `init_global_grid`; it detects the global grid and returns local coordinates with `di` from the global cell counts.
- **Output.** Guard user-facing printing with `igg.me == 0 && …`. Fields written to file carry stale halos unless you `update_halo!` first. `checkpointing_jld2(…, igg)` is rank-aware; `checkpointing_hdf5` is not (give each rank its own `dst`).
- **Column integrals** (`compute_lithostatic_pressure!(P, ρg, dz, igg)`) span ranks along the vertical direction and error if it is periodic.
- **Reporting.** CPU-only validation says nothing about MPI correctness, and a 1-rank run says nothing about halo correctness. When a change touches communication, state the rank counts and backend you actually ran. MPI test conventions are in [testing-rules](testing-rules.md).
