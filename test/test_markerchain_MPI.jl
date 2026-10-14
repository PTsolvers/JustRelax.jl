using Test
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO, JustPIC, PoissonGrids

# Reject whole-field copies and count the corner reads of sparse sampling.
struct SurfaceReadArray{T} <: AbstractMatrix{T}
    data::Matrix{T}
    reads::Base.RefValue{Int}
    fail::Bool
end
Base.size(A::SurfaceReadArray) = size(A.data)
Base.IndexStyle(::Type{<:SurfaceReadArray}) = IndexLinear()
function Base.getindex(A::SurfaceReadArray, i::Int)
    A.fail && error("injected rank-local sample failure")
    A.reads[] += 1
    return A.data[i]
end
Base.similar(::SurfaceReadArray, ::Type{T}, dims::Dims) where {T} = Array{T}(undef, dims)
Base.Array(::SurfaceReadArray) = error("surface sampling must not copy full velocity fields")

function test_markerchain_partition(dims)
    ni = (8, 8)
    igg = IGG(
        init_global_grid(
            ni..., 1; dimx = dims[1], dimy = dims[2],
            init_MPI = !JustRelax.MPI.Initialized(), select_device = false
        )...
    )
    return try
        ng = (nx_g(), ny_g())
        xv = (collect(range(-2.0, 2.0, ng[1] + 1)), collect(range(-1.0, 1.0, ng[2] + 1)) .^ 3)
        global_grid = Geometry(Array, xv...)
        ranges = ntuple(d -> (igg.coords[d] * (ni[d] - 2) + 1):(igg.coords[d] * (ni[d] - 2) + ni[d] + 1), 2)
        grid = Geometry(Array, xv, igg)
        @test_throws DimensionMismatch Geometry(Array, (xv[1][1:(end - 1)], xv[2]), igg)
        monitors = (nothing, window_monitor(2.0, 5.0, 0.25, -0.1))
        refined_global = Geometry(Array, ng, (4.0, 2.0), monitors; origin = (-2.0, -1.0))
        @test_throws DimensionMismatch Geometry(Array, ni .+ 1, (4.0, 2.0), monitors; igg)
        refined = Geometry(Array, ni, (4.0, 2.0), monitors; origin = (-2.0, -1.0), igg)
        @test refined.xvi == ntuple(d -> refined_global.xvi[d][ranges[d]], 2)
        @test refined.li == refined_global.li
        for k in 1:2, d in 1:2
            r = ranges[d]
            indices = first(r):(last(r) + (k != d))
            @test refined.xi_vel[k][d] == refined_global.xi_vel[k][d][indices]
        end
        @test grid.li == global_grid.li
        @test grid.max_li == global_grid.max_li
        @test grid.xvi == ntuple(d -> xv[d][ranges[d]], 2)
        stokes = StokesArrays(CPUBackend, ni)
        for (k, A) in enumerate(@velocity(stokes))
            xi = grid.xi_vel[k]
            @test xi == ntuple(d -> global_grid.xi_vel[k][d][first(ranges[d]):(last(ranges[d]) + (k != d))], 2)
            A .= [0.02 * x - 0.01 * y for x in xi[1], y in xi[2]]

        end
        # Include partition boundaries, physical boundaries, and edge extrapolation.
        points = [(x, y) for x in (-2.1, -2.0, 0.0, 2.0, 2.1) for y in (-1.1, -1.0, 0.0, 1.0, 1.1)]
        active_queries = trues(length(points))
        reads = Ref(0)
        counted = map(A -> SurfaceReadArray(A, reads, false), @velocity(stokes))
        sampled = JustRelax._sample_markerchain_velocity(counted, global_grid.xi_vel, points, active_queries, ni, igg)
        @test sampled[:, 1] ≈ [0.02 * x - 0.01 * y for (x, y) in points]
        @test sampled[:, 2] ≈ sampled[:, 1]
        @test JustRelax.MPI.Allreduce(reads[], +, igg.comm_cart) == 8 * length(points)
        inactive = JustRelax._sample_markerchain_velocity(counted, global_grid.xi_vel, points, falses(length(points)), ni, igg)
        @test all(iszero, inactive)
        failing = map(A -> SurfaceReadArray(A, Ref(0), all(iszero, igg.coords)), @velocity(stokes))
        @test_throws ErrorException JustRelax._sample_markerchain_velocity(failing, global_grid.xi_vel, points, active_queries, ni, igg)

        heights = 0.02 .* cos.(xv[1])
        @test_throws ArgumentError DistributedMarkerChain(
            JustPIC.CPU, xv, igg;
            nxcell = igg.me == 0 ? 0 : 4, min_xcell = 2, max_xcell = 8
        )
        @test_throws ArgumentError DistributedMarkerChain(JustPIC.CPU, xv, igg; initial_elevation = NaN)
        initial = igg.me == 0 ? heights : zero(heights)
        surface = DistributedMarkerChain(
            JustPIC.CPU, xv, igg;
            initial_elevation = initial, nxcell = 4, min_xcell = 2, max_xcell = 8
        )
        chain = surface.chain
        @test surface.local_vertices == ranges[1]
        @test surface.cell_vertices == xv[1]
        @test surface.h_vertices == heights
        @test chain.h_vertices == heights[ranges[1]]
        wrong_stokes = igg.me == 0 ? StokesArrays(CPUBackend, ni .+ 1) : stokes
        @test_throws DimensionMismatch advect_markerchain!(surface, wrong_stokes, 0.01)
        @test surface.h_vertices == heights
        reference = init_markerchain(JustPIC.CPU, 4, 2, 8, xv[1], heights)
        Vglobal = ntuple(k -> [0.02 * x - 0.01 * y for x in global_grid.xi_vel[k][1], y in global_grid.xi_vel[k][2]], 2)
        semilagrangian_advection_markerchain!(reference, RungeKutta2(), Vglobal, global_grid.xi_vel, xv, 0.01)
        advect_markerchain!(surface, stokes, 0.01)
        @test surface.h_vertices ≈ reference.h_vertices
        @test chain.h_vertices ≈ reference.h_vertices[ranges[1]]
        # Nonlinear velocity and surface slopes exercise repeated secant/RK queries,
        # including a surface on and across an internal vertical partition.
        for (initial, surface_dt) in (
                (0.03 .* sin.(pi .* xv[1] ./ 2), 0.5),
                (0.5 .* sin.(6 .* xv[1]), 0.05), (fill(-0.5, length(xv[1])), 0.5),
            )
            for (k, A) in enumerate(@velocity(stokes))
                xi = grid.xi_vel[k]
                A .= [k == 1 ? 0.12 + 0.03 * sin(x) * y : 0.04 + 0.01 * y^2 + 0.02 * x * y for x in xi[1], y in xi[2]]
            end
            Vnonlinear = ntuple(2) do k
                xi = global_grid.xi_vel[k]
                [k == 1 ? 0.12 + 0.03 * sin(x) * y : 0.04 + 0.01 * y^2 + 0.02 * x * y for x in xi[1], y in xi[2]]
            end
            surface.h_vertices .= initial
            reference.h_vertices .= initial
            for _ in 1:3
                semilagrangian_advection_markerchain!(reference, RungeKutta2(), Vnonlinear, global_grid.xi_vel, xv, surface_dt)
                advect_markerchain!(surface, stokes, surface_dt)
                @test surface.h_vertices ≈ reference.h_vertices atol = 1.0e-13 rtol = 1.0e-12
                @test chain.h_vertices ≈ reference.h_vertices[ranges[1]]
                @test JustPIC.mean_height(surface.h_vertices, xv[1]) ≈ JustPIC.mean_height(initial, xv[1]) atol = 1.0e-14
            end
        end
        saved = copy(surface.h_vertices)
        @test_throws ArgumentError advect_markerchain!(
            surface, stokes,
            igg.me == 0 ? NaN : 0.5
        )
        @test surface.h_vertices == saved
        if igg.nprocs > 1
            @test_throws ArgumentError advect_markerchain!(
                surface, stokes,
                igg.me == 0 ? 0.25 : 0.5
            )
            @test surface.h_vertices == saved
        end
        rock = RockRatio(CPUBackend, ni)
        rock_global = RockRatio(CPUBackend, ng)
        compute_rock_fraction!(rock, chain, grid.xvi, grid.di.vertex)
        update_halo!(rock.center, rock.vertex, rock.Vx, rock.Vy)
        compute_rock_fraction!(rock_global, reference, xv, global_grid.di.vertex)
        @test rock.center ≈ rock_global.center[first(ranges[1]):(last(ranges[1]) - 1), first(ranges[2]):(last(ranges[2]) - 1)]

        for field in (:vertex, :Vx, :Vy)
            A, B = getproperty(rock, field), getproperty(rock_global, field)
            indices = ntuple(d -> first(ranges[d]):(first(ranges[d]) + size(A, d) - 1), 2)
            @test A ≈ B[indices...]
        end

        foreach(A -> fill!(A, NaN), @velocity(stokes))
        @test_throws ErrorException advect_markerchain!(surface, stokes, 0.01)
        @test surface.h_vertices ≈ reference.h_vertices

        # Passive identity fields must follow particles through MPI overlap cells.
        particles = init_particles(JustPIC.CPU, 8, 16, 4, grid.xi_vel...)
        px0, py0, tag = init_cell_arrays(particles, Val(3))
        px0.data .= particles.coords[1].data
        py0.data .= particles.coords[2].data
        tag.data .= 7.0
        foreach(A -> fill!(A, 0.1), @velocity(stokes))
        dt = minimum(minimum.(grid.di.vertex))
        dt = JustRelax.JustRelax2D.minimum_mpi(dt)
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        update_cell_halo!(particles.coords..., particles.index, px0, py0, tag)
        move_particles!(particles, (px0, py0, tag))
        active = particles.index.data
        @test any(active)
        @test all(==(7.0), tag.data[active])
        @test particles.coords[1].data[active] ≈ px0.data[active] .+ 0.1 * dt
        @test particles.coords[2].data[active] ≈ py0.data[active] .+ 0.1 * dt

        dst = JustRelax.MPI.bcast(igg.me == 0 ? mktempdir() : nothing, 0, igg.comm_cart)
        V = ntuple(_ -> zeros(ni .+ 1), 2)
        save_pvtk(
            joinpath(dst, "refined"), grid.xvi, grid.xci,
            (; surface = ones(ni .+ 1)), (; rock = rock.center), V, igg;
            t = 0.01, pvd = joinpath(dst, "series")
        )
        JustRelax.MPI.Barrier(igg.comm_cart)
        if igg.me == 0
            header = read(joinpath(dst, "refined.pvtr"), String)
            @test occursin("PRectilinearGrid", header)
            @test occursin("WholeExtent=\"0 $(ng[1]) 0 $(ng[2]) 0 0\"", header)
            @test count("<Piece", header) == igg.nprocs
            @test occursin("Velocity", header)
            @test occursin("refined.pvtr", read(joinpath(dst, "series.pvd"), String))
            @test count(endswith(".vtr"), readdir(joinpath(dst, "refined"))) == igg.nprocs
            rm(dst; recursive = true)
        end
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

JustRelax.MPI.Initialized() || JustRelax.MPI.Init()
nranks = JustRelax.MPI.Comm_size(JustRelax.MPI.COMM_WORLD)
@testset "Distributed marker chain and refined VTK" begin
    test_markerchain_partition((nranks, 1))
    nranks > 1 && test_markerchain_partition((1, nranks))
    nranks == 4 && test_markerchain_partition((2, 2))
end
