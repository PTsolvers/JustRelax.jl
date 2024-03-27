using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

@testset "Flow boundary conditions 2D" begin
    n = 5 # number of elements
    Vx, Vy = PTArray(rand(n + 1, n + 2)), PTArray(rand(n + 2, n + 1))
    
    # free-slip
    bcs = FlowBoundaryConditions(;
        no_slip=(left=false, right=false, top=false, bot=false),
        free_slip=(left=true, right=true, top=true, bot=true),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(bcs, Vx, Vy)

    @test @views Vx[:, 1] == Vx[:, 2]
    @test @views Vx[:, end] == Vx[:, end - 1]
    @test @views Vy[1, :] == Vy[2, :]
    @test @views Vy[end, :] == Vy[end - 1, :]

    # no-slip
    Vx, Vy = PTArray(rand(n + 1, n + 2)), PTArray(rand(n + 2, n + 1))
    bcs = FlowBoundaryConditions(;
        no_slip=(left=true, right=true, top=true, bot=true),
        free_slip=(left=false, right=false, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(bcs, Vx, Vy)
    @test sum(!iszero(Vx[1  , i]) for i in axes(Vx,2)) == 0
    @test sum(!iszero(Vx[end, i]) for i in axes(Vx,2)) == 0
    @test sum(!iszero(Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test sum(!iszero(Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test @views Vy[1  ,   :] == -Vy[2      ,       :]
    @test @views Vy[end,   :] == -Vy[end - 1,       :]
    @test @views Vx[:  ,   1] == -Vx[:      ,       2]
    @test @views Vx[:  , end] == -Vx[:      , end - 1]

    # test with StokesArrays
    # periodicity
    ni = 5, 5
    stokes = StokesArrays(ni, ViscoElastic)
    stokes.V.Vx .= PTArray(rand(n + 1, n + 2))
    stokes.V.Vy .= PTArray(rand(n + 2, n + 1))
    flow_bcs = FlowBoundaryConditions(;
        no_slip=(left=false, right=false, top=false, bot=false),
        free_slip=(left=false, right=false, top=false, bot=false),
        periodicity=(left=true, right=true, top=true, bot=true),
    )
    flow_bcs!(stokes, flow_bcs)

    @test @views stokes.V.Vx[:, 1] == stokes.V.Vx[:, end - 1]
    @test @views stokes.V.Vx[:, end] == stokes.V.Vx[:, 2]
    @test @views stokes.V.Vy[1, :] == stokes.V.Vy[end - 1, :]
    @test @views stokes.V.Vy[end, :] == stokes.V.Vy[2, :]

    # free-slip
    flow_bcs = FlowBoundaryConditions(;
        no_slip=(left=false, right=false, top=false, bot=false),
        free_slip=(left=true, right=true, top=true, bot=true),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(stokes, flow_bcs)

    @test @views stokes.V.Vx[  :,   1] == stokes.V.Vx[      :,       2]
    @test @views stokes.V.Vx[  :, end] == stokes.V.Vx[      :, end - 1]
    @test @views stokes.V.Vy[  1,   :] == stokes.V.Vy[      2,       :]
    @test @views stokes.V.Vy[end,   :] == stokes.V.Vy[end - 1,       :]

    # no-slip
    flow_bcs = FlowBoundaryConditions(;
        no_slip=(left=true, right=true, top=true, bot=true),
        free_slip=(left=false, right=false, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(stokes, flow_bcs)

    @test sum(!iszero(stokes.V.Vx[1  , i]) for i in axes(Vx,2)) == 0
    @test sum(!iszero(stokes.V.Vx[end, i]) for i in axes(Vx,2)) == 0
    @test sum(!iszero(stokes.V.Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test sum(!iszero(stokes.V.Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test @views stokes.V.Vy[1  ,   :] == -stokes.V.Vy[2      ,       :]
    @test @views stokes.V.Vy[end,   :] == -stokes.V.Vy[end - 1,       :]
    @test @views stokes.V.Vx[:  ,   1] == -stokes.V.Vx[:      ,       2]
    @test @views stokes.V.Vx[:  , end] == -stokes.V.Vx[:      , end - 1]
end

@testset "Temperature boundary conditions 2D" begin
    ni      = 5, 5 # number of elements
    thermal = ThermalArrays(ni)
    # free-slip
    bcs = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = true, bot = true),
    )
    thermal_bcs!(thermal, bcs)

    @test @views thermal.T[  :,   1] == thermal.T[      :,       2]
    @test @views thermal.T[  :, end] == thermal.T[      :, end - 1]
    @test @views thermal.T[  1,   :] == thermal.T[      2,       :]
    @test @views thermal.T[end,   :] == thermal.T[end - 1,       :]
end
