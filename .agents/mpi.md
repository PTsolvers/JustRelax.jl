# MPI and distributed runs

Companion to [`AGENTS.md`](../AGENTS.md). Covers domain decomposition, halo exchange,
global reductions, parallel I/O, and how the MPI tests run. Grid semantics are in
[`grid.md`](grid.md), solver structure in [`solvers.md`](solvers.md).

Distributed memory is handled by [ImplicitGlobalGrid.jl][igg], which JustRelax re-exports;
`MPI` is reachable as `JustRelax.MPI`. Every array a solver touches is the **local**
subdomain, extended by the overlap layers ImplicitGlobalGrid allocates.

[igg]: https://github.com/omlins/ImplicitGlobalGrid.jl

## Setting up the topology

```julia
igg = IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
```

`IGG` (`src/grid/Grid.jl`) is a thin container over what `init_global_grid` returns:
`me` (rank), `dims`, `nprocs`, `coords`, `comm_cart`. It is threaded through every solver
call and is what lets solver code find the rank and the Cartesian communicator.

Useful `init_global_grid` keywords: `dimx`/`dimy`/`dimz` to pin the process grid,
`periodx`/`periody`/`periodz` for periodic directions, `init_MPI = false` when MPI is
already initialized, `select_device = false` when several ranks share one GPU or the run is
CPU-only. In a 2D model pass `nz = 1`.

Tear down with `finalize_global_grid(; finalize_MPI = false)` when the process runs further
tests, or with the default when it exits.

`Geometry(ni, li)` detects an initialized global grid automatically
(`ImplicitGlobalGrid.grid_is_initialized()`) and returns coordinates for the local
subdomain, with `di` computed from the global cell counts. Build the grid *after*
`init_global_grid`, never before.

## Halo exchange

`update_halo!(A...)` synchronizes the overlap layers. The solvers call it at the points
where a stencil is about to read a neighbor's data:

- `ητ` (the smoothed PT viscosity) once before the iteration loop and after each viscosity
  update;
- the vertex shear-stress components after the stress update;
- `@velocity(stokes)...` after each velocity update and `flow_bcs!`;
- `thermal.T` after each temperature update and `thermal_bcs!`.

Order matters: boundary conditions are applied first, then the halo is exchanged, so a
rank-interior "boundary" is overwritten by the neighbor's interior values rather than by a
physical BC. A new field that a stencil reads across a rank boundary needs its own
`update_halo!` at the same point in the loop, otherwise the error shows up only under
`mpiexec -n 2` and only near the internal boundaries.

`b_width` (default `(4, 4, 0)` or `(4, 4, 1)` depending on the method) is the halo width for
`@hide_communication b_width begin ... end`, which overlaps the exchange with the interior
computation. Several `@hide_communication` blocks in the sources are commented out; when
re-enabling one, keep the boundary width consistent with the stencil's reach.

## Global reductions

`src/Utils.jl` defines `mean_mpi`, `norm_mpi`, `sum_mpi`, `minimum_mpi`, `maximum_mpi`, all
`MPI.Allreduce` over `MPI.COMM_WORLD`. They are internal (not exported), so reach them as
`JustRelax.JustRelax2D.norm_mpi` if you need them outside the package.

**Any convergence criterion, time-step bound, or diagnostic printed from a solver must go
through these**, never through a bare `norm`/`maximum` on a local array — otherwise ranks
disagree on whether to stop and the run deadlocks or diverges silently.

Residual norms are normalized by the *global* problem size, from `nx_g()`, `ny_g()`,
`nz_g()` (re-exported from ImplicitGlobalGrid) or the internal
`global_grid_size(Val(N))`:

```julia
norm_mpi(stokes.R.RP) / √(nx_g() * ny_g())
```

`compute_dt` has `igg`-aware methods: `compute_dt(stokes, di, dt_diff, igg)` reduces with
`maximum_mpi`, while `compute_dt(stokes, di, dt_diff)` reduces locally. Passing `igg` is not
optional in a distributed run — without it, ranks pick different time steps.

`compute_lithostatic_pressure!(P, ρg, dz, igg)` integrates down a column that spans several
ranks: it builds a Cartesian sub-communicator along the vertical direction with
`MPI.Cart_sub`, `Allgather`s each rank's column contribution, and adds the weight of the
cells held above. It errors if the vertical direction is periodic, since the column
integral is then undefined.

## Output and printing

Guard user-facing output on rank 0:

```julia
igg.me == 0 && @printf(...)
```

The solvers already do this for their convergence logs.

Parallel writers in `JustRelax.DataIO`:

- `save_pvtk(fname, di, data..., igg; t, precision, pvd)` writes one `.vtr` per rank plus a
  `.pvtr` index. Three forms: vertex + center data with velocity, a single group with
  velocity, or a single group alone. It uses the native ImplicitGlobalGrid parallel-VTK API
  (`extents`, `extents_g`, `metagrid`), which requires ImplicitGlobalGrid ≥ 0.17.
- `save_hdf5(fname, dim_g, I, comm_cart, info, data...)` writes a single collective HDF5
  file through the Cartesian communicator.
- `checkpointing_jld2(dst, stokes[, thermal], time, timestep, igg)` names the file after the
  rank (`checkpoint0000.jld2`), so each rank checkpoints its own subdomain; restart with the
  same decomposition and load the file matching the rank. The method without `igg` writes a
  single `checkpoint.jld2` and is for serial runs only.
- `checkpointing_hdf5` always writes `checkpoint.h5` into `dst` — it has no rank-aware
  method, so give each rank a distinct `dst` (or use the JLD2 form) in a distributed run.

Fields written to file carry stale halo values unless you `update_halo!` first.

## Testing

Test files whose name contains `MPI` are pulled out of the parallel test suite by
`test/runtests.jl` and run one at a time under `mpiexec -n 2`:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=test --startup-file=no test/runtests.jl
```

Run one directly while iterating:

```sh
mpiexec -n 2 julia --project=test --startup-file=no test/test_periodic_boundary_conditions_MPI.jl
```

Existing coverage: `test_periodic_boundary_conditions_MPI.jl`,
`test_diffusion2D_multiphase_MPI.jl`, `test_diffusion3D_multiphase_MPI.jl`,
`test_lithostatic_pressure2D_MPI.jl`, `test_lithostatic_pressure3D_MPI.jl`,
`test_shearband2D_MPI.jl`, `test_shearband3D_MPI.jl`, `test_IO_MPI.jl`.

The periodicity test is the pattern to copy for a halo change: it fills each rank's field
with a rank-dependent constant, calls `update_halo!`, and asserts that the boundary rows
now hold the *other* rank's value.

An MPI test must run correctly on a single rank too — `runtests.jl` still invokes it with
two, but the suite is also run interactively. Use `init_MPI = JustRelax.MPI.Initialized() ?
false : true` so the file works both standalone and under a driver that already initialized
MPI.

## Reporting

CPU-only validation says nothing about MPI correctness, and a 1-rank run says nothing about
halo correctness. When a change touches communication, state which rank counts and which
backend you actually exercised.
