# Visualization

JustRelax fields are plain arrays on the staggered grid, so any Julia plotting
package can consume them. Two helpers for the fields that are *not* plain
arrays — particles and cell arrays — ship with the package as a
[Makie.jl](https://docs.makie.org) extension, and become available as soon as a
Makie backend is loaded:

```julia
using JustRelax, JustRelax.JustRelax2D
using CairoMakie   # or GLMakie for interactive windows
```

## Particles

`plot_particles` scatters the particle cloud colored by phase, optionally with a
marker chain drawn on top to show a free surface:

```julia
plot_particles(particles, pPhases; chain = chain, title = "Phases", units = :km)
```

Coordinates are divided by `conversion` (`1.0e3` by default, i.e. meters to
kilometers), so the axes match the `units` label. Passing `filename` writes the
figure to disk in addition to displaying it.

## Cell arrays

Phase ratios and other cell arrays hold several values per grid location, so
they need an index to select one before plotting. `plot_field` takes that index
and the matching coordinates:

```julia
plot_field(phase_ratios.center, 2, xci; title = "Phase 2 fraction")
```

## Grid fields

Velocity, pressure, temperature, stress, and viscosity are ordinary arrays and
go straight to Makie. Velocities live on cell faces, so interpolate them to a
common location first:

```julia
Vx_v = @zeros(ni .+ 1...)
Vy_v = @zeros(ni .+ 1...)
velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

fig = Figure()
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "log10(η)")
heatmap!(ax, xci[1], xci[2], log10.(Array(stokes.viscosity.η)))
arrows2d!(ax, xvi[1][1:5:end], xvi[2][1:5:end], Array(Vx_v)[1:5:end, 1:5:end], Array(Vy_v)[1:5:end, 1:5:end])
fig
```

On a GPU backend, wrap each field in `Array` before handing it to Makie, as
above. The same applies to the grid coordinates when the `Geometry` was built
on the device: `Array.(grid.xci)`. The [Blankenbach](./Blankenbach.md) and
[shear band](./ShearBand2D.md) examples show complete figures built this way.

## ParaView output

For 3D models, large runs, and time series, writing VTK files and inspecting
them in [ParaView](https://www.paraview.org) is usually more practical than
plotting from Julia. `save_vtk` writes the grid fields, `save_particles` and
`save_marker_chain` the particle data; see
[I/O and checkpointing](./api/io.md).

## API reference

```@docs; canonical=false
plot_particles
plot_field
```
