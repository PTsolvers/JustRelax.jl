using Test, Suppressor, JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

@testset "Grid2D" begin
    @suppress begin
        n      = 4 # number of cells
        nx     = n
        ny     = n
        igg    = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
            IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
        else
            igg
        end
        ly     = 1e0         # domain length in y
        lx     = ly          # domain length in x
        ni     = nx, ny      # number of cells
        li     = lx, ly      # domain length in x- and y-
        dx     = lx / nx_g() # grid step in x
        dy     = ly / ny_g() # grid step in y
        di     = dx, dy      # grid step in x- and y
        origin = 0.0, -ly    # origin coordinates (15km f sticky air layer)
        grid   = Geometry(ni, li; origin = origin)

        @test grid.origin == origin
        for i in 1:2
            # test grid at the vertices
            @test grid.xvi[i][1] == origin[i]
            # test grid at the cell centers
            @test grid.xci[i][1] == origin[i] + di[i] / 2
        end
        # test velocity grids
        @test grid.grid_v[1][2][1] == origin[2] - di[1]/2
        @test grid.grid_v[2][1][1] == origin[1] - di[2]/2
    end
end
