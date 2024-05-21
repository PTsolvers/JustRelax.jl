using Test, JustRelax, ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

model = PS_Setup(:cpu, Float64, 3)
environment!(model)

@testset "Flow boundary conditions 3D" begin
    n = 5 # number of elements
    Vx, Vy, Vz = @rand(n + 1, n + 2, n + 2), @rand(n + 2, n + 1, n + 2), @rand(n + 2, n + 2, n + 1)
    # free-slip
    bcs = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        periodicity  = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(bcs, Vx, Vy, Vz)

    # Vx
    @test @views Vx[:, :,   1] == Vx[:, :,     2] # bottom
    @test @views Vx[:, :, end] == Vx[:, :, end-1] # top 
    @test @views Vx[:,   1, :] == Vx[:,     2, :] # left
    @test @views Vx[:, end, :] == Vx[:, end-1, :] # right
    # Vy
    @test @views Vy[:, :,   1] == Vy[:, :,     2] # bottom
    @test @views Vy[:, :, end] == Vy[:, :, end-1] # top 
    @test @views Vy[  1, :, :] == Vy[    2, :, :] # front
    @test @views Vy[end, :, :] == Vy[end-1, :, :] # back
    # Vz
    @test @views Vz[:,   1, :] == Vz[:,     2, :] # left
    @test @views Vz[:, end, :] == Vz[:, end-1, :] # right
    @test @views Vz[  1, :, :] == Vz[    2, :, :] # front
    @test @views Vz[end, :, :] == Vz[end-1, :, :] # back

    # no-slip
    bcs = FlowBoundaryConditions(;
        free_slip   = (left = false, right = false, top = false, bot = false, front = false, back = false),
        no_slip     = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        periodicity = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    Vx, Vy, Vz = @rand(n + 1, n + 2, n + 2), @rand(n + 2, n + 1, n + 2), @rand(n + 2, n + 2, n + 1)
    flow_bcs!(bcs, Vx, Vy, Vz)
    
    # Test the ones that are zero
    @test sum(!iszero(Vx[  1, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # left
    @test sum(!iszero(Vx[end, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # right
    @test sum(!iszero(Vy[i,   1, j]) for i in axes(Vy, 1), j in axes(Vy, 3)) == 0 # front
    @test sum(!iszero(Vy[i, end, j]) for i in axes(Vy, 1), j in axes(Vy, 3)) == 0 # back
    @test sum(!iszero(Vz[i, j,   1]) for i in axes(Vz, 1), j in axes(Vz, 2)) == 0 # bottom
    @test sum(!iszero(Vz[i, j, end]) for i in axes(Vz, 1), j in axes(Vz, 2)) == 0 # top

    # Vx
    @test @views Vx[:,   :,   1] == -Vx[:,     :,     2] # bottom
    @test @views Vx[:,   :, end] == -Vx[:,     :, end-1] # top 
    @test @views Vx[:,   1,   :] == -Vx[:,     2,     :] # left
    @test @views Vx[:, end,   :] == -Vx[:, end-1,     :] # right
    # Vy
    @test @views Vy[  :, :,   1] == -Vy[    :, :,     2] # bottom
    @test @views Vy[  :, :, end] == -Vy[    :, :, end-1] # top 
    @test @views Vy[  1, :,   :] == -Vy[    2, :,     :] # front
    @test @views Vy[end, :,   :] == -Vy[end-1, :,     :] # back
    # Vz
    @test @views Vz[  :,   1, :] == -Vz[    :,     2, :] # left
    @test @views Vz[  :, end, :] == -Vz[    :, end-1, :] # right
    @test @views Vz[  1,   :, :] == -Vz[    2,     :, :] # front
    @test @views Vz[end,   :, :] == -Vz[end-1,     :, :] # back

    # test with StokesArrays
    ni = 5, 5, 5
    stokes = StokesArrays(ni, ViscoElastic)
    # free-slip
    bcs = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = false, front = false, back = false),
        periodicity  = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, bcs)

    # Vx
    @test @views stokes.V.Vx[:, :,   1] == stokes.V.Vx[:, :,     2] # bottom
    @test @views stokes.V.Vx[:, :, end] == stokes.V.Vx[:, :, end-1] # top 
    @test @views stokes.V.Vx[:,   1, :] == stokes.V.Vx[:,     2, :] # left
    @test @views stokes.V.Vx[:, end, :] == stokes.V.Vx[:, end-1, :] # right
    # Vy
    @test @views stokes.V.Vy[:, :,   1] ==stokes.V. Vy[:, :,     2] # bottom
    @test @views stokes.V.Vy[:, :, end] ==stokes.V. Vy[:, :, end-1] # top 
    @test @views stokes.V.Vy[  1, :, :] ==stokes.V. Vy[    2, :, :] # front
    @test @views stokes.V.Vy[end, :, :] ==stokes.V. Vy[end-1, :, :] # back
    # Vz
    @test @views stokes.V.Vz[:,   1, :] == stokes.V.Vz[:,     2, :] # left
    @test @views stokes.V.Vz[:, end, :] == stokes.V.Vz[:, end-1, :] # right
    @test @views stokes.V.Vz[  1, :, :] == stokes.V.Vz[    2, :, :] # front
    @test @views stokes.V.Vz[end, :, :] == stokes.V.Vz[end-1, :, :] # back

    # no-slip
    bcs = FlowBoundaryConditions(;
        free_slip   = (left = false, right = false, top = false, bot = false, front = false, back = false),
        no_slip     = (left = true , right = true , top = true , bot = true , front = true , back = true ),
        periodicity = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, bcs)
    
    # Test the ones that are zero
    @test sum(!iszero(stokes.V.Vx[  1, i, j]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # left
    @test sum(!iszero(stokes.V.Vx[end, i, j]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # right
    @test sum(!iszero(stokes.V.Vy[i,   1, j]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # front
    @test sum(!iszero(stokes.V.Vy[i, end, j]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # back
    @test sum(!iszero(stokes.V.Vz[i, j,   1]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # bottom
    @test sum(!iszero(stokes.V.Vz[i, j, end]) for i in axes(stokes.V.Vx, 2), j in axes(stokes.V.Vx, 3)) == 0 # top

    # Vx
    @test @views stokes.V.Vx[:,   :,   1] == -stokes.V.Vx[:,     :,     2] # bottom
    @test @views stokes.V.Vx[:,   :, end] == -stokes.V.Vx[:,     :, end-1] # top 
    @test @views stokes.V.Vx[:,   1,   :] == -stokes.V.Vx[:,     2,     :] # left
    @test @views stokes.V.Vx[:, end,   :] == -stokes.V.Vx[:, end-1,     :] # right
    # Vy
    @test @views stokes.V.Vy[  :, :,   1] == -stokes.V.Vy[    :, :,     2] # bottom
    @test @views stokes.V.Vy[  :, :, end] == -stokes.V.Vy[    :, :, end-1] # top 
    @test @views stokes.V.Vy[  1, :,   :] == -stokes.V.Vy[    2, :,     :] # front
    @test @views stokes.V.Vy[end, :,   :] == -stokes.V.Vy[end-1, :,     :] # back
    # Vz
    @test @views stokes.V.Vz[  :,   1, :] == -stokes.V.Vz[    :,     2, :] # left
    @test @views stokes.V.Vz[  :, end, :] == -stokes.V.Vz[    :, end-1, :] # right
    @test @views stokes.V.Vz[  1,   :, :] == -stokes.V.Vz[    2,     :, :] # front
    @test @views stokes.V.Vz[end,   :, :] == -stokes.V.Vz[end-1,     :, :] # back

end

@testset "Temperature boundary conditions 3D" begin
    ni      = 5, 5, 5 # number of elements
    thermal = ThermalArrays(ni)
    # free-slip
    bcs = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = true, bot = true, front = true, back = true),
        periodicity = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    thermal_bcs!(thermal, bcs)

    # Vx
    @test @views thermal.T[:, :,   1] == thermal.T[:, :,     2] # bottom
    @test @views thermal.T[:, :, end] == thermal.T[:, :, end-1] # top 
    @test @views thermal.T[:,   1, :] == thermal.T[:,     2, :] # left
    @test @views thermal.T[:, end, :] == thermal.T[:, end-1, :] # right
    # Vy
    @test @views thermal.T[:, :,   1] == thermal.T[:, :,     2] # bottom
    @test @views thermal.T[:, :, end] == thermal.T[:, :, end-1] # top 
    @test @views thermal.T[  1, :, :] == thermal.T[    2, :, :] # front
    @test @views thermal.T[end, :, :] == thermal.T[end-1, :, :] # back
    # Vz
    @test @views thermal.T[:,   1, :] == thermal.T[:,     2, :] # left
    @test @views thermal.T[:, end, :] == thermal.T[:, end-1, :] # right
    @test @views thermal.T[  1, :, :] == thermal.T[    2, :, :] # front
    @test @views thermal.T[end, :, :] == thermal.T[end-1, :, :] # back
end
