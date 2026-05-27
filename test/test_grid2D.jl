@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor, JustRelax, JustRelax.JustRelax2D

@testset "Grid2D" begin
    @suppress begin
        n = 4 # number of cells
        nx = n
        ny = n
        init_mpi = JustRelax.MPI.Initialized() ? false : true
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

        ly = 1.0e0         # domain length in y
        lx = ly          # domain length in x
        ni = nx, ny      # number of cells
        li = lx, ly      # domain length in x- and y-
        dx = lx / nx_g() # grid step in x
        dy = ly / ny_g() # grid step in y
        di = dx, dy      # grid step in x- and y
        origin = 0.0, -ly    # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)

        @test grid.origin == origin
        for i in 1:2
            # test grid at the vertices
            @test grid.xvi[i][1] == origin[i]
            # test grid at the cell centers
            @test grid.xci[i][1] == origin[i] + di[i] / 2
        end
        # test velocity grids
        @test grid.xi_vel[1][2][1] == origin[2] - di[1] / 2
        @test grid.xi_vel[2][1][1] == origin[1] - di[2] / 2

        # nonuniform Geometry(TA, xvi...) constructor + velocity_grids vector path
        xv1 = collect(range(0.0, 1.0; length = 5))
        xv2 = [0.0, 0.4, 0.7, 0.9, 1.0]   # nonuniform along y
        grid_nu = Geometry(Array, xv1, xv2)
        @test grid_nu.ni == (4, 4)
        @test grid_nu.li == (1.0, 1.0)
        @test grid_nu.origin == (0.0, 0.0)
        @test grid_nu.max_li == 1.0
        @test length(grid_nu.xci[1]) == 4 && length(grid_nu.xvi[1]) == 5
        # cell-centered coords are midpoints of supplied vertices
        @test grid_nu.xci[2] == (xv2[1:(end - 1)] .+ xv2[2:end]) ./ 2
        # vector-spacing arrays
        @test grid_nu.di.vertex[2] == diff(xv2)
        @test grid_nu.di.center isa NTuple{2, <:AbstractArray}
        # the staggered velocity grid in the nonuniform direction has one ghost cell
        # on each side, so length = length(xci) + 2
        @test length(grid_nu.xi_vel[1][2]) == length(grid_nu.xci[2]) + 2
        @test length(grid_nu.xi_vel[2][1]) == length(grid_nu.xci[1]) + 2

        # tuple-of-vectors dispatcher: forwards to Geometry(Array, xvi...)
        grid_nu2 = Geometry((xv1, xv2))
        @test grid_nu2.ni == grid_nu.ni
        @test grid_nu2.li == grid_nu.li

        # legacy_uniform_grid: NTuple{N,Real} variant
        leg = JustRelax.legacy_uniform_grid((nx, ny), (dx, dy))
        @test leg.ni == (nx, ny)
        @test leg.li == (lx, ly)

        # legacy_uniform_grid: NamedTuple variant (forwards to .center)
        leg_nt = JustRelax.legacy_uniform_grid((nx, ny), grid.di)
        @test leg_nt.ni == (nx, ny)
        @test leg_nt.li == (lx, ly)

        finalize_global_grid(; finalize_MPI = false)
    end

    @suppress @testset "periodic x_g / y_g" begin
        n = 4
        igg = IGG(
            init_global_grid(
                n, n, 1;
                init_MPI = JustRelax.MPI.Initialized() ? false : true,
                periodx = 1, periody = 1,
            )...,
        )
        dx = 1.0 / n
        # Periodic wrap shifts the first ghost cell to the global left, and
        # values beyond the global span wrap back via the `xi - nx_g()*dx` branch.
        v_first = x_g(1, dx, n)
        v_last = x_g(n + 2, dx, n)   # idx beyond the local span → triggers the wrap
        @test isfinite(v_first) && isfinite(v_last)
        v_first_y = y_g(1, dx, n)
        v_last_y = y_g(n + 2, dx, n)
        @test isfinite(v_first_y) && isfinite(v_last_y)
        finalize_global_grid(; finalize_MPI = false)
    end
end
