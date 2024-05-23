@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustRelax, Test
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

const bk = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    JustRelax.CUDABackend
else
    JustRelax.CPUbackend
end

@testset "Traits" begin
    if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
        # test generic arrays
        @test bk(Array)       === AMDGPUBackendTrait()
        @test bk(Matrix)      === AMDGPUBackendTrait()
        @test bk(Vector)      === AMDGPUBackendTrait()
        @test bk(rand(2))     === AMDGPUBackendTrait()
        @test bk(rand(2,2))   === AMDGPUBackendTrait()
        @test bk(rand(2,2,2)) === AMDGPUBackendTrait()

        # test error handling
        @test_throws ArgumentError bk(rand())
        @test_throws ArgumentError bk("potato")

        # test JR structs
        ## 2D
        ni       = 2, 2
        stokes2  = JR2.StokesArrays(JustRelax.AMDGPUBackend, ni)
        thermal2 = JR2.ThermalArrays(JustRelax.AMDGPUBackend, ni)

        @test bk(stokes2.V) === AMDGPUBackendTrait()
        @test bk(stokes2.τ) === AMDGPUBackendTrait()
        @test bk(stokes2.R) === AMDGPUBackendTrait()
        @test bk(stokes2.P) === AMDGPUBackendTrait()
        @test bk(stokes2)   === AMDGPUBackendTrait()
        @test bk(thermal2)  === AMDGPUBackendTrait()

        ## 3D
        ni       = 2, 2, 2
        stokes3  = JR3.StokesArrays(JustRelax.AMDGPUBackend, ni)
        thermal3 = JR3.ThermalArrays(JustRelax.AMDGPUBackend, ni)

        @test bk(stokes3.V) === AMDGPUBackendTrait()
        @test bk(stokes3.τ) === AMDGPUBackendTrait()
        @test bk(stokes3.R) === AMDGPUBackendTrait()
        @test bk(stokes3.P) === AMDGPUBackendTrait()
        @test bk(stokes3)   === AMDGPUBackendTrait()
        @test bk(thermal3)  === AMDGPUBackendTrait()
    elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
        # test generic arrays
        @test bk(Array)       === CUDABackendTrait()
        @test bk(Matrix)      === CUDABackendTrait()
        @test bk(Vector)      === CUDABackendTrait()
        @test bk(rand(2))     === CUDABackendTrait()
        @test bk(rand(2,2))   === CUDABackendTrait()
        @test bk(rand(2,2,2)) === CUDABackendTrait()

        # test error handling
        @test_throws ArgumentError bk(rand())
        @test_throws ArgumentError bk("potato")

        # test JR structs
        ## 2D
        ni       = 2, 2
        stokes2  = JR2.StokesArrays(JustRelax.CUDABackend, ni)
        thermal2 = JR2.ThermalArrays(JustRelax.CUDABackend, ni)

        @test bk(stokes2.V) === CUDABackendTrait()
        @test bk(stokes2.τ) === CUDABackendTrait()
        @test bk(stokes2.R) === CUDABackendTrait()
        @test bk(stokes2.P) === CUDABackendTrait()
        @test bk(stokes2)   === CUDABackendTrait()
        @test bk(thermal2)  === CUDABackendTrait()

        ## 3D
        ni       = 2, 2, 2
        stokes3  = JR3.StokesArrays(JustRelax.CUDABackend, ni)
        thermal3 = JR3.ThermalArrays(JustRelax.CUDABackend, ni)

        @test bk(stokes3.V) === CUDABackendTrait()
        @test bk(stokes3.τ) === CUDABackendTrait()
        @test bk(stokes3.R) === CUDABackendTrait()
        @test bk(stokes3.P) === CUDABackendTrait()
        @test bk(stokes3)   === CUDABackendTrait()
        @test bk(thermal3)  === CUDABackendTrait()
    else
        # test generic arrays
        @test bk(Array)       === CPUBackendTrait()
        @test bk(Matrix)      === CPUBackendTrait()
        @test bk(Vector)      === CPUBackendTrait()
        @test bk(rand(2))     === CPUBackendTrait()
        @test bk(rand(2,2))   === CPUBackendTrait()
        @test bk(rand(2,2,2)) === CPUBackendTrait()

        # test error handling
        @test_throws ArgumentError bk(rand())
        @test_throws ArgumentError bk("potato")

        # test JR structs
        ## 2D
        ni       = 2, 2
        stokes2  = JR2.StokesArrays(JustRelax.CPUBackend, ni)
        thermal2 = JR2.ThermalArrays(JustRelax.CPUBackend, ni)

        @test bk(stokes2.V) === CPUBackendTrait()
        @test bk(stokes2.τ) === CPUBackendTrait()
        @test bk(stokes2.R) === CPUBackendTrait()
        @test bk(stokes2.P) === CPUBackendTrait()
        @test bk(stokes2)   === CPUBackendTrait()
        @test bk(thermal2)  === CPUBackendTrait()

        ## 3D
        ni       = 2, 2, 2
        stokes3  = JR3.StokesArrays(JustRelax.CPUBackend, ni)
        thermal3 = JR3.ThermalArrays(JustRelax.CPUBackend, ni)

        @test bk(stokes3.V) === CPUBackendTrait()
        @test bk(stokes3.τ) === CPUBackendTrait()
        @test bk(stokes3.R) === CPUBackendTrait()
        @test bk(stokes3.P) === CPUBackendTrait()
        @test bk(stokes3)   === CPUBackendTrait()
        @test bk(thermal3)  === CPUBackendTrait()
    end

end
