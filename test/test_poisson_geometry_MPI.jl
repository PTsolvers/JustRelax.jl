using Test, JustRelax, PoissonGrids

JustRelax.MPI.Initialized() || JustRelax.MPI.Init()
nranks = JustRelax.MPI.Comm_size(JustRelax.MPI.COMM_WORLD)
# Exercise a vertical partition and, on four ranks, a mixed 3D partition.
dims = nranks == 4 ? (2, 1, 2) : (1, 1, nranks)
ni = (6, 6, 6)
igg = IGG(
    init_global_grid(
        ni...; dimx = dims[1], dimy = dims[2], dimz = dims[3],
        init_MPI = false, select_device = false
    )...
)
try
    @testset "PoissonGrids distributed 3D geometry" begin
        monitors = (nothing, nothing, gaussian_monitor(2.0, -0.3, 0.2))
        li, origin = (2.0, 3.0, 1.0), (-1.0, 0.0, -1.0)
        global_grid = Geometry(Array, (nx_g(), ny_g(), nz_g()), li, monitors; origin)
        calls = Ref(0)
        counted_monitor = x -> begin
            calls[] += 1
            igg.me == 0 || error("monitor evaluated away from rank zero")
            monitors[3](x)
        end
        local_grid = Geometry(Array, ni, li, (nothing, nothing, counted_monitor); origin, igg)
        @test (calls[] > 0) == (igg.me == 0)
        @test JustRelax.MPI.Allreduce(Int(calls[] > 0), +, igg.comm_cart) == 1
        calls[] = 0
        # Rank zero's parameters define the one global grid.
        other_li = igg.me == 0 ? li : (9.0, 9.0, 9.0)
        other_origin = igg.me == 0 ? origin : (2.0, 2.0, 2.0)
        vertices = refined_grid_vertices(
            (nx_g(), ny_g(), nz_g()), other_li,
            (nothing, nothing, counted_monitor); origin = other_origin, igg
        )
        @test vertices == global_grid.xvi
        @test (calls[] > 0) == (igg.me == 0)
        @test_throws ArgumentError Geometry(Array, ni, li, (nothing, nothing, x -> -1.0); origin, igg)
        @test_throws PoissonGrids.ConvergenceError refined_grid_vertices(
            (nx_g(), ny_g(), nz_g()), li,
            (nothing, nothing, x -> throw(PoissonGrids.ConvergenceError(3, 1.0, 1.0e-7))); origin, igg
        )
        wrong_ni = igg.me == 0 ? ni .+ 1 : ni
        @test_throws DimensionMismatch Geometry(Array, wrong_ni, li, monitors; origin, igg)
        @test Geometry(Array, ni, li, monitors; origin, igg).xvi == local_grid.xvi
        ranges = ntuple(d -> (igg.coords[d] * (ni[d] - 2) + 1):(igg.coords[d] * (ni[d] - 2) + ni[d] + 1), 3)
        @test local_grid.li == global_grid.li
        @test local_grid.max_li == global_grid.max_li
        @test local_grid.xvi == ntuple(d -> global_grid.xvi[d][ranges[d]], 3)
        for component in 1:3, d in 1:3
            r = ranges[d]
            indices = first(r):(last(r) + (component != d))
            @test local_grid.xi_vel[component][d] == global_grid.xi_vel[component][d][indices]
        end
    end
finally
    finalize_global_grid(; finalize_MPI = false)
end
