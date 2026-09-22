# This provides plotting functionalities using Makie.jl

import JustRelax: plot_particles, plot_field

using JustRelax, JustPIC, CellArrays


function plot_particles(
        particles, pPhases;
        chain = nothing,
        clrmap = :roma,
        title = "Particle Position",
        filename = nothing,
        resolution = (1200, 1200),
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
        save(filename, f)
        display(f)
    else
        display(f)
    end

    return f
end

function plot_field(
        data,
        index::Int,
        grid::NTuple{N, LinRange{T, Int64}};
        colormap = :roma,
        title = "Field Plot",
        filename = nothing,
        resolution = (1200, 1000),
        labelsize = 35,
        titlesize = 50,
        units = :km,
        conversion = 1.0e3

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


    h = heatmap!(ax, x, y, Array(field(data, index)), colormap = colormap)

    Colorbar(f[1, 2], h)
    if !isnothing(filename)
        save(filename, f)
        display(f)
    else
        display(f)
    end

    return f
end
