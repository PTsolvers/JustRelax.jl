# This provides plotting functionalities using Makie.jl

import JustRelax: plot_particles, plot_field

using JustRelax, JustPIC, CellArrays


"""
fig = plot_particles(particles, pPhases;)

2D plot of the particles positions colored by their phase.

# Arguments
- `particles`: Particles from `JustPIC`
- `pPhases`: Field containing the phase information for each particle.

# Keyword Arguments
- `chain`: Markerchain position to e.g. visualise topography (default: `nothing`)
- `clrmap`: Colormap to use (default: `:roma`)
- `title`: Title of the plot (default: `"Particle Position"`)
- `filename`: If provided, saves the figure to this filename (default: `nothing`)
- `resolution`: Resolution of the figure in pixels (default: `(1200, 1200)`)
- `legendsize`: Font size of the legend (default: `15`)
- `labelsize`: Font size of the axis labels (default: `35`)
- `titlesize`: Font size of the title (default: `50`)
- `linecolor`: Color of the markerchain line (default: `:black`)
- `markersize`: Size of the markerchain markers (default: `1.0`)
- `conversion`: Conversion factor for coordinates (default: `1.0e3`, to convert from m to km)
- `units`: Units for the axis labels (default: `:km`)

"""
function plot_particles(
        particles, pPhases;
        chain = nothing,
        clrmap = :roma,
        title = "Particle Position",
        filename = nothing,
        resolution = (1200, 1200),
        legendsize = 15,
        labelsize = 35,
        titlesize = 50,
        linecolor = :black,
        markersize = 1.0,
        conversion = 1.0e3,
        units = :km
    )

    f = Figure(; fontsize = 25, size = resolution)

    # Define axis
    ax = Axis(
        f[1, 1];
        title = title,
        xlabel = "x [$units]",
        ylabel = "y [$units]",
        aspect = DataAspect(),
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        titlesize = titlesize
    )

    p = particles.coords
    ppx, ppy = p
    pxv = ppx.data[:] ./ conversion
    pyv = ppy.data[:] ./ conversion
    clr = pPhases.data[:]
    idxv = particles.index.data[:]

    h = scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = clrmap, markersize = 1)

    if !isnothing(chain)
        chain_x = chain.coords[1].data[:] ./ conversion
        chain_y = chain.coords[2].data[:] ./ conversion
        scatter!(ax, Array(chain_x), Array(chain_y), color = linecolor, markersize = markersize)
    end

    Colorbar(f[1, 2], h)
    if !isnothing(filename)
        save(filename, fig)
        display(f)
    else
        display(f)
    end

    return f
end

"""
    fig = plot_field(field;)

Plots a 2D field from a CellArrays structure using Makie.jl

# Arguments
- `data`: Field to plot, must be a subtype of `CellArrays` e.g. `phase_ratios`
- `grid`: Grid to use for plotting, must be a `LinRange`

# Keyword Arguments
- `clrmap`: Colormap to use (default: `:roma`)
- `title`: Title of the plot (default: `"Field Plot"`)
- `filename`: If provided, saves the figure to this filename (default: `nothing`)
- `resolution`: Resolution of the figure in pixels (default: `(1200, 1200)`)
- `legendsize`: Font size of the legend (default: `15`)
- `labelsize`: Font size of the axis labels (default: `35`)
- `titlesize`: Font size of the title (default: `50`)
- `units`: Units for the axis labels (default: `:km`)

# Example

f = plot_field(phase_ratios.center; title = "Phase Ratios at Cell Centers", units = :km)

"""
function plot_field(
        data, grid::NTuple{N, LinRange{T, Int64}};
        clrmap = :roma,
        title = "Field Plot",
        filename = nothing,
        resolution = (1200, 1200),
        legendsize = 15,
        labelsize = 35,
        titlesize = 50,
        units = :km
    ) where {T, N}

    f = Figure(; fontsize = 25, size = resolution)

    # Define axis
    ax = Axis(
        f[1, 1];
        title = title,
        xlabel = "x [$units]",
        ylabel = "y [$units]",
        aspect = DataAspect(),
        xlabelsize = labelsize,
        ylabelsize = labelsize,
        titlesize = titlesize
    )

    x = grid[1] ./ conversion
    y = grid[2] ./ conversion


    h = heatmap!(ax, x, y, Array(field(data)), colormap = clrmap)

    Colorbar(f[1, 2], h)
    if !isnothing(filename)
        save(filename, fig)
        display(f)
    else
        display(f)
    end

    return f
end
