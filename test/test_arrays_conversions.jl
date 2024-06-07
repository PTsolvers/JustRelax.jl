@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, Test

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

ni      = 2, 2
stokes  = StokesArrays(backend, ni)
thermal = ThermalArrays(backend, ni)

@testset "Type conversions" begin
    A1      = Array(stokes.V)
    A2      = Array(stokes.τ)
    A3      = Array(stokes.R)
    A4      = Array(stokes.P)
    A5      = Array(stokes)
    A6      = Array(thermal)

    @test typeof(A1) <: JustRelax.Velocity{<:Array}
    @test typeof(A2) <: JustRelax.SymmetricTensor{<:Array}
    @test typeof(A3) <: JustRelax.Residual{<:Array}
    @test typeof(A4) <: Array
    @test typeof(A5) <: JustRelax.StokesArrays
    @test typeof(A6) <: JustRelax.ThermalArrays{<:Array}
end

@testset "Type copy" begin
    A1      = copy(stokes.V)
    A2      = copy(stokes.τ)
    A3      = copy(stokes.R)
    A4      = copy(stokes.P)
    A5      = copy(stokes)
    A6      = copy(thermal)

    @test typeof(A1) <: JustRelax.Velocity
    @test typeof(A2) <: JustRelax.SymmetricTensor
    @test typeof(A3) <: JustRelax.Residual
    @test typeof(A4) <: typeof(stokes.P)
    @test typeof(A5) <: JustRelax.StokesArrays
    @test typeof(A6) <: JustRelax.ThermalArrays
end
