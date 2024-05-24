@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, Test
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

const env_backend = ENV["JULIA_JUSTRELAX_BACKEND"]

const backend = @static if env_backend === "AMDGPU"
    AMDGPUBackend
elseif env_backend === "CUDA"
    CUDABackend
else
    CPUBackend
end

A = @static if env_backend === "AMDGPU"
    ROCArray
elseif env_backend === "CUDA"
    CuArray
else
    Array
end

@testset "2D allocators" begin
    ni = nx, ny = (2, 2)

    R = JR2.Residual(ni...)
    @test R isa JustRelax.Residual
    @test isnothing(R.Rz)
    @test size(R.Rx) == (nx-1, ny)
    @test size(R.Ry) == (nx, ny-1)
    @test size(R.RP) == ni
    @test R.Rx isa A
    @test R.Ry isa A
    @test R.RP isa A
    @test_throws MethodError JR2.Residual(10.0, 10.0)

    visc = JR2.Viscosity(ni)
    @test size(visc.η)     == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ)    == ni
    @test visc.η     isa A
    @test visc.η_vep isa A
    @test visc.ητ    isa A
    @test_throws MethodError JR2.Viscosity(10.0, 10.0)

    v      = JR2.Velocity(ni...)
    tensor = JR2.SymmetricTensor(ni...)

    @test size(tensor.xx)   == (nx, ny)
    @test size(tensor.yy)   == (nx, ny)
    @test size(tensor.xy)   == (nx + 1, ny + 1)
    @test size(tensor.xy_c) == (nx, ny)
    @test size(tensor.II)   == (nx, ny)

    @test tensor.xx   isa A
    @test tensor.yy   isa A
    @test tensor.xy   isa A
    @test tensor.xy_c isa A
    @test tensor.II   isa A

    stokes = JR2.StokesArrays(backend, ni)

    @test size(stokes.P)      == ni
    @test size(stokes.P0)     == ni
    @test size(stokes.∇V)     == ni
    @test size(stokes.EII_pl) == ni

    @test stokes.P          isa A
    @test stokes.P0         isa A
    @test stokes.∇V         isa A
    @test stokes.V          isa JustRelax.Velocity
    @test stokes.τ          isa JustRelax.SymmetricTensor
    @test stokes.τ_o        isa JustRelax.SymmetricTensor
    @test stokes.ε          isa JustRelax.SymmetricTensor
    @test stokes.ε_pl       isa JustRelax.SymmetricTensor
    @test stokes.EII_pl     isa A
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
    @test R.Rx isa A
    @test R.Ry isa A
    @test R.Rz isa A
    @test R.RP isa A
    @test_throws MethodError JR3.Residual(1.0, 1.0, 1.0)

    visc = JR3.Viscosity(ni)
    @test size(visc.η)     == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ)    == ni
    @test visc.η     isa A
    @test visc.η_vep isa A
    @test visc.ητ    isa A
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

    @test tensor.xx   isa A
    @test tensor.yy   isa A
    @test tensor.xy   isa A
    @test tensor.yz   isa A
    @test tensor.xz   isa A
    @test tensor.xy_c isa A
    @test tensor.yz_c isa A
    @test tensor.xz_c isa A
    @test tensor.II   isa A

    stokes = JR3.StokesArrays(backend, ni)

    @test size(stokes.P)      == ni
    @test size(stokes.P0)     == ni
    @test size(stokes.∇V)     == ni
    @test size(stokes.EII_pl) == ni

    @test stokes.P          isa A
    @test stokes.P0         isa A
    @test stokes.∇V         isa A
    @test stokes.V          isa JustRelax.Velocity
    @test stokes.τ          isa JustRelax.SymmetricTensor
    @test stokes.τ_o        isa JustRelax.SymmetricTensor
    @test stokes.ε          isa JustRelax.SymmetricTensor
    @test stokes.ε_pl       isa JustRelax.SymmetricTensor
    @test stokes.EII_pl     isa A
    @test stokes.viscosity  isa JustRelax.Viscosity
    @test stokes.R          isa JustRelax.Residual

    @test_throws MethodError JR3.StokesArrays(backend, 10.0, 10.0)
end
