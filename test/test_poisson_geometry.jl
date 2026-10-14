using Test
using JustRelax

using PoissonGrids

@testset "PoissonGrids geometry extension" begin
    @test !isnothing(Base.get_extension(JustRelax, :JustRelaxPoissonGridsExt))
    ni, li, origin = (18, 16), (450.0, 210.0), (-225.0, -200.0)
    M = window_monitor(5.0, 0.25, 32.5, -27.5)
    grid = Geometry(ni, li, (nothing, M); origin)
    xv = (
        collect(range(origin[1], origin[1] + li[1], ni[1] + 1)),
        solve_grid(origin[2], origin[2] + li[2], M, ni[2]),
    )
    @test refined_grid_vertices(ni, li, (nothing, M); origin) == xv
    reference = Geometry(xv)
    @test grid.xvi == reference.xvi
    @test grid.xci == reference.xci
    @test grid.xi_vel == reference.xi_vel
    @test grid.di == reference.di
    @test grid._di == reference._di
    @test grid.ni == ni
    @test grid.li == li
    @test grid.origin == origin
    @test grid.max_li == maximum(li)

    # The same constructor builds 3D coordinates; solver support is unchanged.
    grid3 = Geometry(
        Array, (6, 8, 10), (1.0, 2.0, 3.0),
        (nothing, gaussian_monitor(2.0, 1.0, 0.3), nothing)
    )
    @test length.(grid3.xvi) == (7, 9, 11)
    @test grid3.xvi[2] == solve_grid(0.0, 2.0, gaussian_monitor(2.0, 1.0, 0.3), 8)
    @test grid3.di.vertex[2] == diff(grid3.xvi[2])
    @test_throws ArgumentError Geometry((1, 8), (1.0, 1.0), (nothing, nothing))
    @test_throws ArgumentError Geometry((8, 8), (-1.0, 1.0), (nothing, nothing))
    @test_throws ArgumentError Geometry((8, 8), (1.0, 1.0), (nothing, nothing); origin = (NaN, 0.0))
    @test_throws ArgumentError Geometry((8, 8), (1.0, 1.0), (x -> -1.0, nothing))
end
