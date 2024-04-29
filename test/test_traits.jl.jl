using JustRelax, Test
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

const bk = JustRelax.backend

@testset "Traits" begin
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
    stokes2  = JR2.StokesArrays(CPUBackend, ni)
    thermal2 = JR2.ThermalArrays(CPUBackend, ni)

    @test bk(stokes2.V) === CPUBackendTrait()
    @test bk(stokes2.τ) === CPUBackendTrait()
    @test bk(stokes2.R) === CPUBackendTrait()
    @test bk(stokes2.P) === CPUBackendTrait()
    @test bk(stokes2)   === CPUBackendTrait()
    @test bk(thermal2)  === CPUBackendTrait()
    
    ## 3D
    ni       = 2, 2, 2
    stokes3  = JR3.StokesArrays(CPUBackend, ni)
    thermal3 = JR3.ThermalArrays(CPUBackend, ni)
 
    @test bk(stokes3.V) === CPUBackendTrait()
    @test bk(stokes3.τ) === CPUBackendTrait()
    @test bk(stokes3.R) === CPUBackendTrait()
    @test bk(stokes3.P) === CPUBackendTrait()
    @test bk(stokes3)   === CPUBackendTrait()
    @test bk(thermal3)  === CPUBackendTrait()
end