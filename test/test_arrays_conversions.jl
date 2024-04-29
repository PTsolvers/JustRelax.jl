# using JustRelax, JustRelax.JustRelax2D, Test
# const bk = JustRelax.backend

# # @testset "Array conversions" begin
#     ni      = 2, 2
#     stokes  = StokesArrays(CPUBackend, ni)
#     thermal = ThermalArrays(CPUBackend, ni)

#     # @test Array(stokes.V) isa Velocity{Array{T, N}} where {T, N}
#     # @test Array(stokes.τ) isa SymmetricTensor{Array{T, N}} where {T, N}
#     # @test Array(stokes.R) isa Residual{Array{T, N}} where {T, N}
#     # @test Array(stokes.P) isa Array{T, N} where {T, N}
#     # @test Array(stokes)   isa StokesArrays
#     # @test Array(thermal)  isa ThermalArrays{Array{T, N}} where {T, N}
    

#     # test generic arrays
#     @test bk(Array)       === CPUBackendTrait()
#     @test bk(Matrix)      === CPUBackendTrait()
#     @test bk(Vector)      === CPUBackendTrait()
#     @test bk(rand(2))     === CPUBackendTrait()
#     @test bk(rand(2,2))   === CPUBackendTrait()
#     @test bk(rand(2,2,2)) === CPUBackendTrait()
#     @test_throws ArgumentError backend(rand()) 

#     # test JR structs
#     @test bk(stokes.V) === CPUBackendTrait()
#     @test bk(stokes.τ) === CPUBackendTrait()
#     @test bk(stokes.R) === CPUBackendTrait()
#     @test bk(stokes.P) === CPUBackendTrait()
#     @test bk(stokes)   === CPUBackendTrait()
#     @test bk(thermal)  === CPUBackendTrait()
#     @test_throws ArgumentError bk("potato")
# # end

