using Test, Suppressor, JustRelax, JustRelax.JustRelax3D

@testset "Grid3D" begin
    @suppress begin
        n            = 4 # number of cells
        nx = ny = nz = n
        igg          = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
            IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
        else
            igg
        end
        lx = ly = lz = 1e0   # domain length in x
        ni           = nx, ny, nz     # number of cells
        li           = lx, ly, lz     # domain length in x- and y-
        dx           = lx / nx_g()    # grid step in x
        dy           = ly / ny_g()    # grid step in y
        dz           = lz / nz_g()    # grid step in y
        di           = dx, dy, dz     # grid step in x- and y
        origin       = 0.0, 0.0, -lz  # origin coordinates (15km f sticky air layer)
        grid         = Geometry(ni, li; origin = origin)

        @test grid.origin == origin
        for i in 1:3
            # test grid at the vertices
            @test grid.xvi[i][1] == origin[i]
            # test grid at the cell centers
            @test grid.xci[i][1] == origin[i] + di[i] / 2
        end
        # test velocity grids
        @test grid.grid_v[1][2][1] == origin[2] - di[1]/2
        @test grid.grid_v[2][1][1] == origin[1] - di[2]/2
        @test grid.grid_v[3][1][1] == origin[1] - di[3]/2
    end
end
