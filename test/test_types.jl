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

const BackendArray = PTArray(backend)

@testset "2D allocators" begin
    ni = nx, ny = (2, 2)

    stokes = JR2.StokesArrays(backend, ni)

    @test size(stokes.P) == ni
    @test size(stokes.P0) == ni
    @test size(stokes.∇V) == ni
    @test size(stokes.EII_pl) == ni

    @test typeof(stokes.P) <: BackendArray
    @test typeof(stokes.P0) <: BackendArray
    @test typeof(stokes.∇V) <: BackendArray
    @test stokes.V isa JustRelax.Velocity
    @test stokes.U isa JustRelax.Displacement
    @test stokes.ω isa JustRelax.Vorticity
    @test stokes.τ isa JustRelax.SymmetricTensor
    @test stokes.τ_o isa JustRelax.SymmetricTensor
    @test stokes.ε isa JustRelax.SymmetricTensor
    @test stokes.ε_pl isa JustRelax.SymmetricTensor
    @test typeof(stokes.EII_pl) <: BackendArray
    @test stokes.viscosity isa JustRelax.Viscosity
    @test stokes.R isa JustRelax.Residual

    R = stokes.R
    @test R isa JustRelax.Residual
    @test isnothing(R.Rz)
    @test size(R.Rx) == (nx - 1, ny)
    @test size(R.Ry) == (nx, ny - 1)
    @test size(R.RP) == ni
    @test typeof(R.Rx) <: BackendArray
    @test typeof(R.Ry) <: BackendArray
    @test typeof(R.RP) <: BackendArray
    @test_throws MethodError JR2.Residual(10.0, 10.0)

    visc = stokes.viscosity
    @test size(visc.η) == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ) == ni
    @test typeof(visc.η) <: BackendArray
    @test typeof(visc.η_vep) <: BackendArray
    @test typeof(visc.ητ) <: BackendArray
    @test_throws MethodError JR2.Viscosity(10.0, 10.0)

    tensor = stokes.τ

    @test size(tensor.xx) == (nx, ny)
    @test size(tensor.yy) == (nx, ny)
    @test size(tensor.xy) == (nx + 1, ny + 1)
    @test size(tensor.xy_c) == (nx, ny)
    @test size(tensor.II) == (nx, ny)

    @test typeof(tensor.xx) <: BackendArray
    @test typeof(tensor.yy) <: BackendArray
    @test typeof(tensor.xy) <: BackendArray
    @test typeof(tensor.xy_c) <: BackendArray
    @test typeof(tensor.II) <: BackendArray

    @test_throws MethodError JR2.StokesArrays(backend, 10.0, 10.0)
    @test_throws MethodError JR2.Velocity(10.0, 10.0)
    @test_throws MethodError JR2.Displacement(10.0, 10.0)
    @test_throws MethodError JR2.Vorticity(10.0, 10.0)
    @test_throws MethodError JR2.SymmetricTensor(10.0, 10.0, 10.0)

    σ = JR2.PrincipalStress(backend, ni)

    @test size(σ.σ1) == (2, ni...)
    @test size(σ.σ2) == (2, ni...)
    @test size(σ.σ3) == (2, 1, 1)
    @test JR2.compute_principal_stresses!(stokes, σ) == nothing

    @test_throws MethodError JR2.PrincipalStress(backend, 10.0, 10.0)

    thermal = JR2.ThermalArrays(backend, ni)
    @test size(thermal.T) == (nx + 3, ny + 1)
    @test size(thermal.Tc) == ni
    @test size(thermal.Told) == (nx + 3, ny + 1)
    @test size(thermal.ΔT) == (nx + 3, ny + 1)
    @test size(thermal.ΔTc) == ni
    @test size(thermal.adiabatic) == (nx + 1, ny - 1)
    @test size(thermal.dT_dt) == (nx + 1, ny - 1)
    @test size(thermal.qTx) == (nx + 2, ny - 1)
    @test size(thermal.qTy) == (nx + 1, ny)
    @test size(thermal.qTx2) == (nx + 2, ny - 1)
    @test size(thermal.qTy2) == (nx + 1, ny)
    @test size(thermal.ResT) == (nx + 1, ny - 1)
    @test thermal.qTz === nothing
    @test thermal.qTz2 === nothing

    @test typeof(thermal.T) <: BackendArray
    @test typeof(thermal.Tc) <: BackendArray
    @test typeof(thermal.Told) <: BackendArray
    @test typeof(thermal.ΔT) <: BackendArray
    @test typeof(thermal.ΔTc) <: BackendArray
    @test typeof(thermal.adiabatic) <: BackendArray
    @test typeof(thermal.dT_dt) <: BackendArray
    @test typeof(thermal.qTx) <: BackendArray
    @test typeof(thermal.qTy) <: BackendArray
    @test typeof(thermal.qTx2) <: BackendArray
    @test typeof(thermal.qTy2) <: BackendArray
    @test typeof(thermal.ResT) <: BackendArray

    @test JR2.ThermalArrays(10, 10) isa JustRelax.ThermalArrays
    @test JR2.ThermalArrays(ni...) isa JustRelax.ThermalArrays

    @test_throws MethodError JR2.ThermalArrays(10.0, 10.0)

end

@testset "2D Displacement" begin
    ni = nx, ny = (2, 2)
    stokes = JR2.StokesArrays(backend, ni)

    stokes.V.Vx .= 1.0
    stokes.V.Vy .= 1.0

    JR2.velocity2displacement!(stokes, 10)
    @test all(stokes.U.Ux .== 10)

    JR2.displacement2velocity!(stokes, 5)
    @test all(stokes.V.Vx .== 2.0)
end

@testset "3D allocators" begin
    ni = nx, ny, nz = (2, 2, 2)

    stokes = JR3.StokesArrays(backend, ni)

    @test size(stokes.P) == ni
    @test size(stokes.P0) == ni
    @test size(stokes.∇V) == ni
    @test size(stokes.EII_pl) == ni

    @test typeof(stokes.P) <: BackendArray
    @test typeof(stokes.P0) <: BackendArray
    @test typeof(stokes.∇V) <: BackendArray
    @test stokes.V isa JustRelax.Velocity
    @test stokes.U isa JustRelax.Displacement
    @test stokes.ω isa JustRelax.Vorticity
    @test stokes.τ isa JustRelax.SymmetricTensor
    @test stokes.τ_o isa JustRelax.SymmetricTensor
    @test stokes.ε isa JustRelax.SymmetricTensor
    @test stokes.ε_pl isa JustRelax.SymmetricTensor
    @test typeof(stokes.EII_pl) <: BackendArray
    @test stokes.viscosity isa JustRelax.Viscosity
    @test stokes.R isa JustRelax.Residual

    R = stokes.R
    @test R isa JustRelax.Residual
    @test size(R.Rx) == (nx - 1, ny, nz)
    @test size(R.Ry) == (nx, ny - 1, nz)
    @test size(R.Rz) == (nx, ny, nz - 1)
    @test size(R.RP) == ni
    @test typeof(R.Rx) <: BackendArray
    @test typeof(R.Ry) <: BackendArray
    @test typeof(R.Rz) <: BackendArray
    @test typeof(R.RP) <: BackendArray
    @test_throws MethodError JR3.Residual(1.0, 1.0, 1.0)

    visc = stokes.viscosity
    @test size(visc.η) == ni
    @test size(visc.η_vep) == ni
    @test size(visc.ητ) == ni
    @test typeof(visc.η) <: BackendArray
    @test typeof(visc.η_vep) <: BackendArray
    @test typeof(visc.ητ) <: BackendArray
    @test_throws MethodError JR3.Viscosity(1.0, 1.0, 1.0)

    tensor = stokes.τ

    @test size(tensor.xx) == ni
    @test size(tensor.yy) == ni
    @test size(tensor.xy) == (nx + 1, ny + 1, nz)
    @test size(tensor.yz) == (nx, ny + 1, nz + 1)
    @test size(tensor.xz) == (nx + 1, ny, nz + 1)
    @test size(tensor.xy_c) == ni
    @test size(tensor.yz_c) == ni
    @test size(tensor.xz_c) == ni
    @test size(tensor.II) == ni

    @test typeof(tensor.xx) <: BackendArray
    @test typeof(tensor.yy) <: BackendArray
    @test typeof(tensor.xy) <: BackendArray
    @test typeof(tensor.yz) <: BackendArray
    @test typeof(tensor.xz) <: BackendArray
    @test typeof(tensor.xy_c) <: BackendArray
    @test typeof(tensor.yz_c) <: BackendArray
    @test typeof(tensor.xz_c) <: BackendArray
    @test typeof(tensor.II) <: BackendArray

    @test_throws MethodError JR3.StokesArrays(backend, 10.0, 10.0, 10.0)
    @test_throws MethodError JR3.Velocity(10.0, 10.0, 10.0)
    @test_throws MethodError JR3.Displacement(10.0, 10.0, 10.0)
    @test_throws MethodError JR3.Vorticity(10.0, 10.0, 10.0)
    @test_throws MethodError JR3.SymmetricTensor(10.0, 10.0, 10.0)


    σ = JR3.PrincipalStress(backend, ni)
    @test size(σ.σ1) == (3, ni...)
    @test size(σ.σ2) == (3, ni...)
    @test size(σ.σ3) == (3, ni...)
    @test JR3.compute_principal_stresses!(stokes, σ) == nothing

    thermal = JR3.ThermalArrays(backend, ni)
    @test size(thermal.T) == (nx + 1, ny + 1, nz + 1)
    @test size(thermal.Tc) == ni
    @test size(thermal.Told) == (nx + 1, ny + 1, nz + 1)
    @test size(thermal.ΔT) == (nx + 1, ny + 1, nz + 1)
    @test size(thermal.ΔTc) == ni
    @test size(thermal.adiabatic) == (nx - 1, ny - 1, nz - 1)
    @test size(thermal.dT_dt) == (nx - 1, ny - 1, nz - 1)
    @test size(thermal.qTx) == (nx, ny - 1, nz - 1)
    @test size(thermal.qTy) == (nx - 1, ny, nz - 1)
    @test size(thermal.qTz) == (nx - 1, ny - 1, nz)
    @test size(thermal.qTx2) == (nx, ny - 1, nz - 1)
    @test size(thermal.qTy2) == (nx - 1, ny, nz - 1)
    @test size(thermal.qTz2) == (nx - 1, ny - 1, nz)
    @test size(thermal.ResT) == (nx - 1, ny - 1, nz - 1)

    @test typeof(thermal.T) <: BackendArray
    @test typeof(thermal.Tc) <: BackendArray
    @test typeof(thermal.Told) <: BackendArray
    @test typeof(thermal.ΔT) <: BackendArray
    @test typeof(thermal.ΔTc) <: BackendArray
    @test typeof(thermal.adiabatic) <: BackendArray
    @test typeof(thermal.dT_dt) <: BackendArray
    @test typeof(thermal.qTx) <: BackendArray
    @test typeof(thermal.qTy) <: BackendArray
    @test typeof(thermal.qTz) <: BackendArray
    @test typeof(thermal.qTx2) <: BackendArray
    @test typeof(thermal.qTy2) <: BackendArray
    @test typeof(thermal.qTz2) <: BackendArray
    @test typeof(thermal.ResT) <: BackendArray
    @test JR3.ThermalArrays(10, 10, 10) isa JustRelax.ThermalArrays
    @test JR3.ThermalArrays(ni...) isa JustRelax.ThermalArrays

    @test_throws MethodError JR3.ThermalArrays(10.0, 10.0, 10.0)
end

@testset "3D Displacement" begin
    ni = nx, ny, nz = (2, 2, 2)
    stokes = JR3.StokesArrays(backend, ni)

    stokes.V.Vx .= 1.0
    stokes.V.Vy .= 1.0
    stokes.V.Vz .= 1.0

    JR3.velocity2displacement!(stokes, 10)
    @test all(stokes.U.Ux .== 10.0)

    JR3.displacement2velocity!(stokes, 5)
    @test all(stokes.V.Vx .== 2.0)
end
