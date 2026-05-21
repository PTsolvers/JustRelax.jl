@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor, JustRelax, JustRelax.JustRelax3D

@testset "Grid3D" begin
    @suppress begin
        n = 4 # number of cells
        nx = ny = nz = n
        init_mpi = JustRelax.MPI.Initialized() ? false : true
        igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)
        lx = ly = lz = 1.0e0   # domain length in x
        ni = nx, ny, nz     # number of cells
        li = lx, ly, lz     # domain length in x- and y-
        dx = lx / nx_g()    # grid step in x
        dy = ly / ny_g()    # grid step in y
        dz = lz / nz_g()    # grid step in y
        di = dx, dy, dz     # grid step in x- and y
        origin = 0.0, 0.0, -lz  # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)

        @test grid.origin == origin
        for i in 1:3
            # test grid at the vertices
            @test grid.xvi[i][1] == origin[i]
            # test grid at the cell centers
            @test grid.xci[i][1] == origin[i] + di[i] / 2
        end
        # test velocity grids
        @test grid.xi_vel[1][2][1] == origin[2] - di[1] / 2
        @test grid.xi_vel[2][1][1] == origin[1] - di[2] / 2
        @test grid.xi_vel[3][1][1] == origin[1] - di[3] / 2

        # nonuniform Geometry(TA, xvi...) constructor + velocity_grids vector path (3D)
        xv1 = collect(range(0.0, 1.0; length = 5))
        xv2 = [0.0, 0.4, 0.7, 0.9, 1.0]   # nonuniform along y
        xv3 = collect(range(0.0, 2.0; length = 5))
        grid_nu = Geometry(Array, xv1, xv2, xv3)
        @test grid_nu.ni == (4, 4, 4)
        @test grid_nu.li == (1.0, 1.0, 2.0)
        @test grid_nu.origin == (0.0, 0.0, 0.0)
        @test grid_nu.max_li == 2.0
        @test grid_nu.xci[2] == (xv2[1:(end - 1)] .+ xv2[2:end]) ./ 2
        @test grid_nu.di.vertex[2] == diff(xv2)
        @test length(grid_nu.xi_vel) == 3
        # ghost cells extend the transverse directions by 2 in 3D as well
        @test length(grid_nu.xi_vel[1][2]) == length(grid_nu.xci[2]) + 2
        @test length(grid_nu.xi_vel[3][1]) == length(grid_nu.xci[1]) + 2

        # tuple-of-vectors dispatcher: forwards to Geometry(Array, xvi...)
        grid_nu2 = Geometry((xv1, xv2, xv3))
        @test grid_nu2.ni == grid_nu.ni
        @test grid_nu2.li == grid_nu.li

        # legacy_uniform_grid: 3D NTuple and NamedTuple variants
        leg = JustRelax.legacy_uniform_grid((nx, ny, nz), (dx, dy, dz))
        @test leg.ni == (nx, ny, nz)
        @test leg.li == (lx, ly, lz)

        leg_nt = JustRelax.legacy_uniform_grid((nx, ny, nz), grid.di)
        @test leg_nt.ni == (nx, ny, nz)
        @test leg_nt.li == (lx, ly, lz)

        finalize_global_grid(; finalize_MPI = false)
    end

    @suppress @testset "periodic z_g" begin
        n = 4
        igg = IGG(
            init_global_grid(
                n, n, n;
                init_MPI = JustRelax.MPI.Initialized() ? false : true,
                periodx = 1, periody = 1, periodz = 1,
            )...,
        )
        dx = 1.0 / n
        v_first = z_g(1, dx, n)
        v_last = z_g(n + 2, dx, n)
        @test isfinite(v_first) && isfinite(v_last)
        finalize_global_grid(; finalize_MPI = true)
    end
end
