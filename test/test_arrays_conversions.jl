using JustRelax, Test
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

@testset "Array conversions" begin
    ni = 10, 10
    stokes  = StokesArrays(ni, ViscoElastic)
    thermal = ThermalArrays(ni)

    @test Array(stokes.V) isa Velocity{Array{T, N}} where {T, N}
    @test Array(stokes.τ) isa SymmetricTensor{Array{T, N}} where {T, N}
    @test Array(stokes.R) isa Residual{Array{T, N}} where {T, N}
    @test Array(stokes.P) isa Array{T, N} where {T, N}
    @test Array(stokes)   isa StokesArrays
    @test Array(thermal)  isa ThermalArrays{Array{T, N}} where {T, N}
    
    @test JustRelax.iscpu(stokes.V) isa JustRelax.CPUDeviceTrait
    @test JustRelax.iscpu(stokes.τ) isa JustRelax.CPUDeviceTrait
    @test JustRelax.iscpu(stokes.R) isa JustRelax.CPUDeviceTrait
    @test JustRelax.iscpu(stokes.P) isa JustRelax.CPUDeviceTrait
    @test JustRelax.iscpu(stokes)   isa JustRelax.CPUDeviceTrait
    @test JustRelax.iscpu(thermal)  isa JustRelax.CPUDeviceTrait
    @test_throws ArgumentError("Unkown device") JustRelax.iscpu("potato")
end
