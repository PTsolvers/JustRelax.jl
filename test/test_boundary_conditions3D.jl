using Test, JustRelax, ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# @testset begin
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
    flow_bcs!(bcs, Vx, Vy, Vz)
    
    # Test the ones that are zero
    @test sum(!iszero(Vx[  1, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # left
    @test sum(!iszero(Vx[end, i, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # right
    @test sum(!iszero(Vy[i,   1, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # front
    @test sum(!iszero(Vy[i, end, j]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # back
    @test sum(!iszero(Vz[i, j,   1]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # lefbottom
    @test sum(!iszero(Vz[i, j, end]) for i in axes(Vx, 2), j in axes(Vx, 3)) == 0 # trop

    ## TODO: everything below here
    @test sum(!iszero(Vx[end, i]) for i in axes(Vx,2)) == 0
    @test sum(!iszero(Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test sum(!iszero(Vy[i,   1]) for i in axes(Vy,1)) == 0
    @test @views Vy[1, :]   == -Vy[2, :]
    @test @views Vy[end, :] == -Vy[end - 1, :]
    @test @views Vx[:, 1]   == -Vx[:, 2]
    @test @views Vx[:, end] == -Vx[:, end - 1]

    # test with StokesArrays
    ni = 5, 5
    stokes = StokesArrays(ni, ViscoElastic)
    # free-slip
    flow_bcs = FlowBoundaryConditions(;
        no_slip=(left=false, right=false, top=false, bot=false),
        free_slip=(left=true, right=true, top=true, bot=true),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(stokes, flow_bcs)

    @test @views stokes.V.Vx[:, 1] == stokes.V.Vx[:, 2]
    @test @views stokes.V.Vx[:, end] == stokes.V.Vx[:, end - 1]
    @test @views stokes.V.Vy[1, :] == stokes.V.Vy[2, :]
    @test @views stokes.V.Vy[end, :] == stokes.V.Vy[end - 1, :]

    # no-slip
    flow_bcs = FlowBoundaryConditions(;
        no_slip=(left=true, right=true, top=true, bot=true),
        free_slip=(left=false, right=false, top=false, bot=false),
        periodicity=(left=false, right=false, top=false, bot=false),
    )
    flow_bcs!(stokes, flow_bcs)

    @test @views stokes.V.Vx[1, :] ==
        stokes.V.Vx[end, :] ==
        stokes.V.Vy[:, 1] ==
        stokes.V.Vy[:, end]
    @test @views stokes.V.Vx[:, 2] == stokes.V.Vx[:, 3] / 3
    @test @views stokes.V.Vx[:, end-1] == stokes.V.Vx[:, end - 2] / 3
    @test @views stokes.V.Vy[2, :] == stokes.V.Vy[3, :] / 3
    @test @views stokes.V.Vy[end-1, :] == stokes.V.Vy[end - 2, :] / 3
# end


# function free_slip!(Ax, Ay, Az, bc)
#     n1, n2 = bc_index((Ax,Ay,Az))

#     for i in 1:n1, j in 1:n2
#         # free slip in the top and bottom XY planes
#         if bc.top
#             if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
#                 Ax[i, j, 1] = Ax[i, j, 2]
#             end
#             if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
#                 Ay[i, j, 1] = Ay[i, j, 2]
#             end
#         end
#         if bc.bot
#             if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
#                 Ax[i, j, end] = Ax[i, j, end - 1]
#             end
#             if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
#                 Ay[i, j, end] = Ay[i, j, end - 1]
#             end
#         end
#     end
#     return nothing
# end

V = Vx, Vy, Vz



@parallel_indices (i, j) function foo!(Ax, Ay, Az, bc)
        
        # free slip in the top and bottom XY planes
        if bc.top
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, 1] = Ax[i, j, 2]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, 1] = Ay[i, j, 2]
            end
        end
        if bc.bot
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, end] = Ax[i, j, end - 1]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, end] = Ay[i, j, end - 1]
            end
        end
        # # free slip in the front and back XZ planes
        if bc.front
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, 1, j] = Ax[i, 2, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, 1, j] = Az[i, 2, j]
            end
        end
        # if bc.back
        #     if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
        #         Ax[i, end, j] = Ax[i, end - 1, j]
        #     end
        #     if i ≤ size(Az, 1) && j ≤ size(Az, 3)
        #         Az[i, end, j] = Az[i, end - 1, j]
        #     end
        # end
        # # free slip in the front and back YZ planes
        # if bc.left
        #     if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
        #         Ay[1, i, j] = Ay[2, i, j]
        #     end
        #     if i ≤ size(Az, 2) && j ≤ size(Az, 3)
        #         Az[1, i, j] = Az[2, i, j]
        #     end
        # end
        # if bc.right
        #     if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
        #         Ay[end, i, j] = Ay[end - 1, i, j]
        #     end
        #     if i ≤ size(Az, 2) && j ≤ size(Az, 3)
        #         Az[end, i, j] = Az[end - 1, i, j]
        #     end
        # end
    return nothing
end

# free_slip!(Vx, Vy, Vz, bcs.free_slip)
n = 5
V = Vx, Vy, Vz = @rand(n + 1, n + 2, n + 2), @rand(n + 2, n + 1, n + 2), @rand(n + 2, n + 2, n + 1)

nn = bc_index(V)
@parallel (@idx nn) foo!(Vx, Vy, Vz, bcs.free_slip)

Vx[:, :, 1] .== Vx[:, :, 2]
Vx[:, :, end] .== Vx[:, :, end-1]