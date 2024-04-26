using JustRelax, Test
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

@testset "Array conversions" begin
    ni = 10, 10
    stokes  = StokesArrays(ni, ViscoElastic)
    thermal = ThermalArrays(ni)

    @test Array(stokes.V) isa Velocity{Matrix{Float64}}
    @test Array(stokes.τ) isa SymmetricTensor{Matrix{Float64}}
    @test Array(stokes.R) isa Residual{Matrix{Float64}}
    @test Array(stokes.P) isa Matrix
    @test Array(stokes)   isa StokesArrays
    @test Array(thermal)  isa ThermalArrays{Matrix{Float64}}
end
