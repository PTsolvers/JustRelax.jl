# Field advection

## Particles-in-Cell
[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) relies on [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) for advections of particles containing material information.

The recommended workflow is now:

```julia
using JustRelax
using JustPIC, JustPIC._2D

grid = Geometry(ni, li; origin = origin)

nxcell = 24
max_xcell = 36
min_xcell = 12

particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid.xi_vel...)
pT, pPhases = init_cell_arrays(particles, Val(2))
particle_args = (pT, pPhases)
phase_ratios = PhaseRatios(backend, nphases, ni)
```

`grid.xi_vel` stores the staggered velocity coordinates used to initialize particles. In most cases you should pass `grid.xi_vel...` directly to `init_particles` instead of rebuilding the velocity grids manually from `xci`, `xvi`, and `di`.

## Typical particle operations

Common particle operations now follow the compact API used in the tests and examples:

```julia
grid2particle!(pT, T_buffer, particles)
particle2grid!(T_buffer, pT, particles)

advection!(particles, RungeKutta2(), @velocity(stokes), dt)
move_particles!(particles, particle_args)
inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,))
update_phase_ratios!(phase_ratios, particles, pPhases)
```

If you use subgrid diffusion, the matching workflow is:

```julia
subgrid_arrays = SubgridDiffusionCellArrays(particles)
dt₀ = similar(stokes.P)

subgrid_characteristic_time!(
    subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
)
centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
subgrid_diffusion!(pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, dt)
```

## Velocity grids

`velocity_grids(xci, xvi, di)` is still available when you need the staggered coordinates explicitly, for example for analysis or custom utilities. When you already have a [`Geometry`](@ref), prefer `grid.xi_vel`.
