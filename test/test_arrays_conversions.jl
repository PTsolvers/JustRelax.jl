@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using JustRelax, JustRelax.JustRelax2D, Test

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@testset "Array conversions" begin
    ni      = 2, 2
    stokes  = StokesArrays(backend, ni)
    thermal = ThermalArrays(backend, ni)

    @test Array(stokes.V) isa JustRelax.Velocity{Array{T, N}} where {T, N}
    @test Array(stokes.τ) isa JustRelax.SymmetricTensor{Array{T, N}} where {T, N}
    @test Array(stokes.R) isa JustRelax.Residual{Array{T, N}} where {T, N}
    @test Array(stokes.P) isa Array{T, N} where {T, N}
    @test Array(stokes)   isa JustRelax.StokesArrays
    @test Array(thermal)  isa JustRelax.ThermalArrays{Array{T, N}} where {T, N}
end
