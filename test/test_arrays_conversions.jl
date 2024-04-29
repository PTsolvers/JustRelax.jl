using JustRelax, JustRelax.JustRelax2D, Test
const bk = JustRelax.backend

@testset "Array conversions" begin
    ni      = 2, 2
    stokes  = StokesArrays(CPUBackend, ni)
    thermal = ThermalArrays(CPUBackend, ni)
    
    @test Array(stokes.V) isa JustRelax.Velocity{Array{T, N}} where {T, N}
    @test Array(stokes.τ) isa JustRelax.SymmetricTensor{Array{T, N}} where {T, N}
    @test Array(stokes.R) isa JustRelax.Residual{Array{T, N}} where {T, N}
    @test Array(stokes.P) isa Array{T, N} where {T, N}
    @test Array(stokes)   isa JustRelax.StokesArrays
    @test Array(thermal)  isa JustRelax.ThermalArrays{Array{T, N}} where {T, N}    
end

