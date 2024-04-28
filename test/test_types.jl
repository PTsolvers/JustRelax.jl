using JustRelax, Test
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

const backend = CPUBackend

@testset "2D allocators" begin
    ni = nx, ny = (2, 2)

    R = JR2.Residual(ni...)
    @test R isa JustRelax.Residual
    @test isnothing(R.Rz)
    @test size(R.Rx) == (nx-1, ny)
    @test size(R.Ry) == (nx, ny-1)
    @test size(R.RP) == ni
    @test R.Rx isa Array
    @test R.Ry isa Array
    @test R.RP isa Array
    @test_throws MethodError JR2.Residual(10.0, 10.0)

    visc = JR2.Viscosity(ni)
    @test size(visc.η)     == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ)    == ni
    @test visc.η     isa Array
    @test visc.η_vep isa Array
    @test visc.ητ    isa Array
    @test_throws MethodError JR2.Viscosity(10.0, 10.0)

    v      = JR2.Velocity(ni...)
    tensor = JR2.SymmetricTensor(ni...)

    @test size(tensor.xx)   == (nx, ny)
    @test size(tensor.yy)   == (nx, ny)
    @test size(tensor.xy)   == (nx + 1, ny + 1)
    @test size(tensor.xy_c) == (nx, ny)
    @test size(tensor.II)   == (nx, ny)

    @test tensor.xx   isa Array
    @test tensor.yy   isa Array
    @test tensor.xy   isa Array
    @test tensor.xy_c isa Array
    @test tensor.II   isa Array

    stokes = JR2.StokesArrays(backend, ni)

    @test size(stokes.P)      == ni
    @test size(stokes.P0)     == ni
    @test size(stokes.∇V)     == ni
    @test size(stokes.EII_pl) == ni

    @test stokes.P          isa Array
    @test stokes.P0         isa Array
    @test stokes.∇V         isa Array
    @test stokes.V          isa JustRelax.Velocity
    @test stokes.τ          isa JustRelax.SymmetricTensor
    @test stokes.τ_o        isa JustRelax.SymmetricTensor
    @test stokes.ε          isa JustRelax.SymmetricTensor
    @test stokes.ε_pl       isa JustRelax.SymmetricTensor
    @test stokes.EII_pl     isa Array
    @test stokes.viscosity  isa JustRelax.Viscosity
    @test stokes.R          isa JustRelax.Residual

    @test_throws MethodError JR2.StokesArrays(backend, 10.0, 10.0)
end

@testset "3D allocators" begin
    ni = nx, ny, nz = (2, 2, 2)
    
    R = JR3.Residual(ni...)
    @test R isa JustRelax.Residual
    @test size(R.Rx) == (nx-1, ny, nz) 
    @test size(R.Ry) == (nx, ny-1, nz)
    @test size(R.Rz) == (nx, ny, nz-1)
    @test size(R.RP) == ni
    @test R.Rx isa Array
    @test R.Ry isa Array
    @test R.Rz isa Array
    @test R.RP isa Array
    @test_throws MethodError JR3.Residual(1.0, 1.0, 1.0)

    visc = JR3.Viscosity(ni)
    @test size(visc.η)     == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ)    == ni
    @test visc.η     isa Array
    @test visc.η_vep isa Array
    @test visc.ητ    isa Array
    @test_throws MethodError JR3.Viscosity(1.0, 1.0, 1.0)

    v      = JR3.Velocity(ni...)
    tensor = JR3.SymmetricTensor(ni...)

    @test size(tensor.xx)   == ni
    @test size(tensor.yy)   == ni
    @test size(tensor.xy)   == (nx + 1, ny + 1, nz    )
    @test size(tensor.yz)   == (nx    , ny + 1, nz + 1)
    @test size(tensor.xz)   == (nx + 1, ny    , nz + 1)
    @test size(tensor.xy_c) == ni
    @test size(tensor.yz_c) == ni
    @test size(tensor.xz_c) == ni
    @test size(tensor.II)   == ni

    @test tensor.xx   isa Array
    @test tensor.yy   isa Array
    @test tensor.xy   isa Array
    @test tensor.yz   isa Array
    @test tensor.xz   isa Array
    @test tensor.xy_c isa Array
    @test tensor.yz_c isa Array
    @test tensor.xz_c isa Array
    @test tensor.II   isa Array

    stokes = JR3.StokesArrays(backend, ni)

    @test size(stokes.P)      == ni
    @test size(stokes.P0)     == ni
    @test size(stokes.∇V)     == ni
    @test size(stokes.EII_pl) == ni

    @test stokes.P          isa Array
    @test stokes.P0         isa Array
    @test stokes.∇V         isa Array
    @test stokes.V          isa JustRelax.Velocity
    @test stokes.τ          isa JustRelax.SymmetricTensor
    @test stokes.τ_o        isa JustRelax.SymmetricTensor
    @test stokes.ε          isa JustRelax.SymmetricTensor
    @test stokes.ε_pl       isa JustRelax.SymmetricTensor
    @test stokes.EII_pl     isa Array
    @test stokes.viscosity  isa JustRelax.Viscosity
    @test stokes.R          isa JustRelax.Residual

    @test_throws MethodError JR3.StokesArrays(backend, 10.0, 10.0)
end