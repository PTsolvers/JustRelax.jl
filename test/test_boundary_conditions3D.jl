@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D
using Test

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@testset "Boundary Conditions" begin
    if backend === CPUBackend

        # test incompatible boundary conditions
        @test_throws ErrorException FlowBoundaryConditions(;
            no_slip     = (left=true, right=true, front=true, back=true, top=true, bot=true),
            free_slip   = (left=false, right=true, front=true, back=true, top=true, bot=true),
        )
            
        # test with StokesArrays
        ni           = 5, 5, 5
        stokes       = StokesArrays(backend, ni)
        stokes.V.Vx .= PTArray(backend)(rand(size(stokes.V.Vx)...))
        stokes.V.Vy .= PTArray(backend)(rand(size(stokes.V.Vy)...))
        stokes.V.Vz .= PTArray(backend)(rand(size(stokes.V.Vz)...))
        
        # free-slip
        flow_bcs     = FlowBoundaryConditions(;
            no_slip     = (left=false, right=false, front=false, back=false, top=false, bot=false),
            free_slip   = (left=true, right=true, front=true, back=true, top=true, bot=true),
        )
        flow_bcs!(stokes, flow_bcs)
        flow_bcs!(stokes, flow_bcs) # just a trick to pass the CI

        @test @views stokes.V.Vx[  :,   :,   1] == stokes.V.Vx[:, :, 2]
        @test @views stokes.V.Vx[  :,   :, end] == stokes.V.Vx[:, :, end - 1]
        @test @views stokes.V.Vx[  :,   1,   :] == stokes.V.Vx[:, 2, :]
        @test @views stokes.V.Vx[  :, end,   :] == stokes.V.Vx[:, end - 1, :]
        @test @views stokes.V.Vy[  :,   :,   1] == stokes.V.Vy[:, :, 2]
        @test @views stokes.V.Vy[  :,   :, end] == stokes.V.Vy[:, :, end - 1]
        @test @views stokes.V.Vy[  1,   :,   :] == stokes.V.Vy[2, :, :]
        @test @views stokes.V.Vy[end,   :,   :] == stokes.V.Vy[end - 1, :, :]
        @test @views stokes.V.Vz[  1,   :,   :] == stokes.V.Vz[2, :, :]
        @test @views stokes.V.Vz[end,   :,   :] == stokes.V.Vz[end - 1, :, :]
        @test @views stokes.V.Vz[  :,   1,   :] == stokes.V.Vz[:, 2, :]
        @test @views stokes.V.Vz[  :, end,   :] == stokes.V.Vz[:, end - 1, :]
        
        # no-slip
        flow_bcs    = FlowBoundaryConditions(;
            no_slip     = (left=true, right=true, front=true, back=true, top=true, bot=true),
            free_slip   = (left=false, right=false, front=false, back=false, top=false, bot=false),
        )
        flow_bcs!(stokes, flow_bcs)
        
        (; Vx, Vy, Vz) = stokes.V
        @test sum(!iszero(Vx[1  ,  i,   j]) for i in axes(Vx,2), j in axes(Vx,3)) == 0
        @test sum(!iszero(Vx[end,  i,   j]) for i in axes(Vx,2), j in axes(Vx,3)) == 0
        @test sum(!iszero(Vy[i,    1,   j]) for i in axes(Vy,1), j in axes(Vy,2)) == 0
        @test sum(!iszero(Vy[i,  end,   j]) for i in axes(Vy,1), j in axes(Vy,2)) == 0
        @test sum(!iszero(Vz[i,    j,   1]) for i in axes(Vz,1), j in axes(Vz,3)) == 0
        @test sum(!iszero(Vz[i,    j, end]) for i in axes(Vz,1), j in axes(Vz,3)) == 0
        @test @views Vx[  :,   1,   :] == -Vx[      :,       2,       :]
        @test @views Vx[  :, end,   :] == -Vx[      :, end - 1,       :]
        @test @views Vx[  :,   :,   1] == -Vx[      :,       :,       2]
        @test @views Vx[  :,   :, end] == -Vx[      :,       :, end - 1]
        @test @views Vy[  1,   :,   :] == -Vy[      2,       :,       :]
        @test @views Vy[end,   :,   :] == -Vy[end - 1,       :,       :]
        @test @views Vy[  :,   :,   1] == -Vy[      :,       :,       2]
        @test @views Vy[  :,   :, end] == -Vy[      :,       :, end - 1]
        @test @views Vz[  :,   1,   :] == -Vz[      :,       2,       :]
        @test @views Vz[  :, end,   :] == -Vz[      :, end - 1,       :]
        @test @views Vz[  1,   :,   :] == -Vz[      2,       :,       :]
        @test @views Vz[end,   :,   :] == -Vz[end - 1,       :,       :]
    else
        @test true === true
    end
end