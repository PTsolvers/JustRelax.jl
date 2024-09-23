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
    A7      = Array(stokes.U)
    A8      = Array(stokes.ω)

    @test typeof(A1) <: JustRelax.Velocity{<:Array}
    @test typeof(A2) <: JustRelax.SymmetricTensor{<:Array}
    @test typeof(A3) <: JustRelax.Residual{<:Array}
    @test typeof(A4) <: Array
    @test typeof(A5) <: JustRelax.StokesArrays
    @test typeof(A6) <: JustRelax.ThermalArrays{<:Array}
    @test typeof(A7) <: JustRelax.Displacement{<:Array}
    @test typeof(A8) <: JustRelax.Vorticity{<:Array}
end

@testset "Type copy" begin
    S1      = copy(stokes.V)
    S2      = copy(stokes.τ)
    S3      = copy(stokes.R)
    S4      = copy(stokes.P)
    S5      = copy(stokes)
    S6      = copy(stokes.U)
    S7      = copy(stokes.ω)
    T1      = copy(thermal)
        
    @test typeof(S1) <: JustRelax.Velocity
    @test typeof(S2) <: JustRelax.SymmetricTensor
    @test typeof(S3) <: JustRelax.Residual
    @test typeof(S4) <: typeof(stokes.P)
    @test typeof(S5) <: JustRelax.StokesArrays
    @test typeof(S6) <: JustRelax.Displacement
    @test typeof(S7) <: JustRelax.Vorticity
    @test typeof(T1) <: JustRelax.ThermalArrays
end
