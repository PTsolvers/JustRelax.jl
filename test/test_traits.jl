@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, Test
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3
import JustRelax: AMDGPUBackendTrait, CUDABackendTrait

const bk = JustRelax.backend

const env_backend = ENV["JULIA_JUSTRELAX_BACKEND"]

const DeviceTrait = @static if env_backend === "AMDGPU"
    AMDGPUBackendTrait
elseif env_backend === "CUDA"
    CUDABackendTrait
else
    CPUBackendTrait
end

const backend = @static if env_backend === "AMDGPU"
    AMDGPUBackend
elseif env_backend === "CUDA"
    CUDABackend
else
    CPUBackend
end

const myrand = @static if env_backend === "AMDGPU"
    AMDGPU.rand
elseif env_backend === "CUDA"
    CUDA.rand
else
    rand
end

A, M, V = @static if env_backend === "AMDGPU"
    RocArray, RocMatrix, RocVector
elseif env_backend === "CUDA"
    CuArray, CuMatrix, CuVector
else
    Array, Matrix, Vector
end

@testset "Traits" begin
    # test generic arrays
    @test bk(A)             === DeviceTrait()
    @test bk(M)             === DeviceTrait()
    @test bk(V)             === DeviceTrait()
    @test bk(myrand(2))     === DeviceTrait()
    @test bk(myrand(2,2))   === DeviceTrait()
    @test bk(myrand(2,2,2)) === DeviceTrait()

    # test error handling
    @test_throws ArgumentError bk(myrand())
    @test_throws ArgumentError bk("potato")

    # test JR structs
    ## 2D
    ni       = 2, 2
    stokes2  = JR2.StokesArrays(backend, ni)
    thermal2 = JR2.ThermalArrays(backend, ni)

    @test bk(stokes2.V) === DeviceTrait()
    @test bk(stokes2.τ) === DeviceTrait()
    @test bk(stokes2.R) === DeviceTrait()
    @test bk(stokes2.P) === DeviceTrait()
    @test bk(stokes2)   === DeviceTrait()
    @test bk(thermal2)  === DeviceTrait()

    ## 3D
    ni       = 2, 2, 2
    stokes3  = JR3.StokesArrays(backend, ni)
    thermal3 = JR3.ThermalArrays(backend, ni)

    @test bk(stokes3.V) === DeviceTrait()
    @test bk(stokes3.τ) === DeviceTrait()
    @test bk(stokes3.R) === DeviceTrait()
    @test bk(stokes3.P) === DeviceTrait()
    @test bk(stokes3)   === DeviceTrait()
    @test bk(thermal3)  === DeviceTrait()
end
