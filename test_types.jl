using Test
using ParallelStencil
@init_parallel_stencil(Threads,Float64,2)

include("src/stokes/types/types.jl")

T = Array

ni = nx, ny = (10, 10)

R = Residual(ni)
@test isnothing(R.Rz)
@test size(R.Rx) == (nx-1, ny)
@test size(R.Ry) == (nx, ny-1)
@test size(R.RP) == ni
@test R.Rx isa Array
@test R.Ry isa Array
@test R.RP isa Array

@test Residual(ni)     isa Residual
@test Residual(nx, ny) isa Residual
@test_throws ArgumentError Residual(10.0, 10.0)

visc = Viscosity(ni)
@test size(visc.η)     == ni
@test size(visc.η_vep) == ni
@test size(visc.ητ)    == ni
@test visc.η     isa Array
@test visc.η_vep isa Array
@test visc.ητ    isa Array
@test_throws ArgumentError Viscosity(10.0, 10.0)

v = Velocity(ni)
tensor = SymmetricTensor(ni)

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

stokes = StokesArrays(ni)

@test size(stokes.P)      == ni
@test size(stokes.P0)     == ni
@test size(stokes.∇V)     == ni
@test size(stokes.EII_pl) == ni

@test stokes.P          isa Array
@test stokes.P0         isa Array
@test stokes.∇V         isa Array
@test stokes.V          isa Velocity
@test stokes.τ          isa SymmetricTensor
@test stokes.τ_o        isa SymmetricTensor
@test stokes.ε          isa SymmetricTensor
@test stokes.ε_pl       isa SymmetricTensor
@test stokes.EII_pl     isa Array
@test stokes.viscosity  isa Viscosity
@test stokes.R          isa Residual

@test_throws ArgumentError StokesArrays(10.0, 10.0)
