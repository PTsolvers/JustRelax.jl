@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax2D: locate_row_index

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

@parallel_indices (i, j) function init_phases!(phases, py, index, y_surf, air_phase)
    for ip in cellaxes(phases)
        @index(index[ip, i, j]) || continue
        y = @index py[ip, i, j]
        @index phases[ip, i, j] = y > y_surf ? Float64(air_phase) : 1.0
    end
    return nothing
end

@testset "Topography correction on a refined grid" begin
    nx, ny = 32, 40
    lx, ly = 100.0e3, 60.0e3
    origin = 0.0, -ly
    # cell height grows downwards, from 37.5 m at the surface to ~3 km at the bottom
    xv = collect(LinRange(0.0, lx, nx + 1))
    s = LinRange(0, 1, ny + 1)
    yv = @. origin[2] + ly * s^2
    grid = Geometry(Array, xv, yv)
    dyv = grid.di.vertex[2]

    @testset "locate_row_index" begin
        # row j spans yv[j] ≤ y < yv[j+1]
        reference(y) = clamp(searchsortedlast(yv, y), 1, ny)
        ys = vcat(
            collect(grid.xci[2]),
            collect(yv[1:(end - 1)]) .+ 1.0e-6,
            collect(yv[2:end]) .- 1.0e-6,
            origin[2] .+ ly .* rand(500),
        )
        @test all(locate_row_index(y, origin[2], dyv) == reference(y) for y in ys)
        # out of the domain clamps to the first/last row
        @test locate_row_index(origin[2] - 1.0e3, origin[2], dyv) == 1
        @test locate_row_index(1.0e3, origin[2], dyv) == ny
        # a constant spacing vector and the equivalent scalar agree
        dy = ly / ny
        ys_uniform = origin[2] .+ ly .* rand(500)
        @test all(
            locate_row_index(y, origin[2], dy) ==
                locate_row_index(y, origin[2], fill(dy, ny)) for y in ys_uniform
        )
    end

    @testset "update_phases_given_markerchain!" begin
        nxcell, max_xcell, min_xcell = 12, 24, 6
        particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid.xi_vel...)
        pPhases, = init_cell_arrays(particles, Val(1))
        air_phase = 2

        y_surf = -0.05 * ly
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, grid.xvi[1], y_surf)
        @parallel (@idx size(particles.index)) init_phases!(
            pPhases, particles.coords[2], particles.index, y_surf, air_phase
        )

        # raise the surface: the air particles left underneath it must be deleted
        y_new = -0.02 * ly
        fill!(chain.coords[2].data, y_new)
        update_phases_given_markerchain!(
            pPhases, chain, particles, origin, grid.di.vertex, air_phase
        )

        index, py, ph = Array(particles.index), Array(particles.coords[2]), Array(pPhases)
        stale = 0
        for j in axes(index, 2), i in axes(index, 1)
            for ip in cellaxes(index)
                @index(index[ip, i, j]) || continue
                y = @index py[ip, i, j]
                phase = @index ph[ip, i, j]
                # air below the new surface, or rock above it, is a cell that was
                # never visited because its row fell outside the searched range
                ((phase == air_phase) != (y > y_new)) && (stale += 1)
            end
        end
        @test stale == 0
    end
end
