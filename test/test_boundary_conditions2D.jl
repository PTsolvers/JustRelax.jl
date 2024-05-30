@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Test, Suppressor

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@testset "Boundary Conditions" begin
    if backend === CPUBackend
        @suppress begin

            # test incompatible boundary conditions
            @test_throws ErrorException FlowBoundaryConditions(;
                no_slip     = (left=false, right=false, top=false, bot=false),
                free_slip   = (left=false, right=true, top=true, bot=true),
            )
            @test_throws ErrorException FlowBoundaryConditions(;
                no_slip     = (left=false, right=false, top=false, bot=false),
                free_slip   = (left=true , right=true , top=true , bot=false),
            )

            n       = 5 # number of elements
            Vx, Vy  = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
            # free-slip
            bcs     = FlowBoundaryConditions(;
                no_slip     = (left=false, right=false, top=false, bot=false),
                free_slip   = (left=true, right=true, top=true, bot=true),
            )
            flow_bcs!(bcs, Vx, Vy)

            @test @views Vx[:  ,   1] == Vx[:,       2]
            @test @views Vx[:  , end] == Vx[:, end - 1]
            @test @views Vy[1  ,   :] == Vy[2,       :]
            @test @views Vy[end,   :] == Vy[end - 1, :]

            # no-slip
            Vx, Vy  = PTArray(backend)(rand(n + 1, n + 2)), PTArray(backend)(rand(n + 2, n + 1))
            bcs     = FlowBoundaryConditions(;
                no_slip     = (left=true, right=true, top=true, bot=true),
                free_slip   = (left=false, right=false, top=false, bot=false),
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
            ni           = 5, 5
            stokes       = StokesArrays(backend, ni)
            stokes.V.Vx .= PTArray(backend)(rand(n + 1, n + 2))
            stokes.V.Vy .= PTArray(backend)(rand(n + 2, n + 1))
            # free-slip
            flow_bcs     = FlowBoundaryConditions(;
                no_slip     = (left=false, right=false, top=false, bot=false),
                free_slip   = (left=true, right=true, top=true, bot=true),
            )
            flow_bcs!(stokes, flow_bcs)

            @test @views stokes.V.Vx[:,   1] == stokes.V.Vx[:,       2]
            @test @views stokes.V.Vx[:, end] == stokes.V.Vx[:, end - 1]
            @test @views stokes.V.Vy[1,   :] == stokes.V.Vy[2,       :]
            @test @views stokes.V.Vy[end, :] == stokes.V.Vy[end - 1, :]

            # no-slip
            flow_bcs    = FlowBoundaryConditions(;
                no_slip     = (left=true, right=true, top=true, bot=true),
                free_slip   = (left=false, right=false, top=false, bot=false),
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
    else
        @test true === true
    end
end
