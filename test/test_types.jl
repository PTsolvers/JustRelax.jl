@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using JustRelax, Test
using GeoParams
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

    # Non-uniform vertex shear and xx ≠ -yy, so both the 4-vertex mean of
    # squares and the plane-strain zz = -(xx + yy) term enter the invariant.
    εxy_v = [Float64(i + 2j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    stokes.ε_pl.xx .= 3.0
    stokes.ε_pl.yy .= -1.0
    copyto!(stokes.ε_pl.xy, εxy_v)
    εII_ref = [
        sqrt(0.5 * (3.0^2 + 1.0^2 + 2.0^2) + sum(abs2, εxy_v[i:(i + 1), j:(j + 1)]) / 4)
            for i in 1:nx, j in 1:ny
    ]
    stokes.EII_pl .= 1.0
    JR2.accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, 2.0)
    @test Array(stokes.EII_pl) ≈ 1.0 .+ 2.0 .* εII_ref

    JR2.tensor_invariant!(stokes.ε_pl)
    @test Array(stokes.ε_pl.II) ≈ εII_ref

    stokes.EVol_pl .= 1.0
    stokes.ε_vol_pl .= -0.25
    JR2.accumulate_vol!(stokes.EVol_pl, stokes.ε_vol_pl, 2.0)
    @test all(Array(stokes.EVol_pl) .≈ 0.5)

    thermal = JR2.ThermalArrays(backend, ni)
    @test size(thermal.T) == (nx + 2, ny + 2)
    @test size(@view(thermal.T[2:(end - 1), 2:(end - 1)])) == ni
    @test parent(@view(thermal.T[2:(end - 1), 2:(end - 1)])) === thermal.T
    @test size(thermal.Told) == (nx + 2, ny + 2)
    @test size(thermal.ΔT) == (nx + 2, ny + 2)
    @test size(thermal.adiabatic) == ni
    @test size(thermal.dT_dt) == ni
    @test size(thermal.qTx) == (nx + 1, ny)
    @test size(thermal.qTy) == (nx, ny + 1)
    @test size(thermal.qTx2) == (nx + 1, ny)
    @test size(thermal.qTy2) == (nx, ny + 1)
    @test size(thermal.ResT) == ni
    @test thermal.qTz === nothing
    @test thermal.qTz2 === nothing

    @test typeof(thermal.T) <: BackendArray
    @test typeof(thermal.Told) <: BackendArray
    @test typeof(thermal.ΔT) <: BackendArray
    @test typeof(thermal.adiabatic) <: BackendArray
    @test typeof(thermal.dT_dt) <: BackendArray
    @test typeof(thermal.qTx) <: BackendArray
    @test typeof(thermal.qTy) <: BackendArray
    @test typeof(thermal.qTx2) <: BackendArray
    @test typeof(thermal.qTy2) <: BackendArray
    @test typeof(thermal.ResT) <: BackendArray

    elastic = ConstantElasticity(; G = 10.0, Kb = 20.0)
    rheology = SetMaterialParams(;
        Phase = 1,
        ShearHeat = ConstantShearheating(; Χ = 1.0),
        Elasticity = elastic,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 2.0), elastic)),
    )
    stokes.τ.xx .= 2.0
    stokes.τ.yy .= -2.0
    stokes.τ.xy_c .= 3.0
    stokes.τ_o.xx .= 0.0
    stokes.τ_o.yy .= 0.0
    stokes.τ_o.xy_c .= 0.0
    # Incompressible strain rate (εzz = 0) with non-uniform vertex shear, which the
    # kernel averages onto centers. ε_el = (τ - τ_o) / (2 G dt) and
    # H = Χ τij (εij - ε_el,ij), with the xy term counted twice.
    εxy_v = [0.25 * (i + 2j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    stokes.ε.xx .= 0.5
    stokes.ε.yy .= -0.5
    copyto!(stokes.ε.xy, εxy_v)
    G, dt = 10.0, 2.0
    H_ref = [
        2.0 * (0.5 - 2.0 / (2G * dt)) + (-2.0) * (-0.5 + 2.0 / (2G * dt)) +
            2 * 3.0 * (sum(εxy_v[i:(i + 1), j:(j + 1)]) / 4 - 3.0 / (2G * dt))
            for i in 1:nx, j in 1:ny
    ]
    JR2.compute_shear_heating!(thermal, stokes, rheology, dt)
    @test Array(thermal.shear_heating) ≈ H_ref

    thermal_rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2700.0),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1000.0),
            Conductivity = ConstantConductivity(; k = 3.0),
        ),
    )
    dt₀ = similar(thermal.T)
    fill!(dt₀, 0.0)
    phases = PTArray(backend)(fill(1, ni...))
    # dt₀ = ρCp / (2k (1/dx² + 1/dy²)); anisotropic spacing exposes a swapped axis.
    JR2.subgrid_characteristic_time!(
        nothing, nothing, dt₀, phases, thermal_rheology, thermal, stokes, (1.0, 2.0)
    )
    @test all(@view(Array(dt₀)[2:(end - 1), 2:(end - 1)]) .≈ 2700.0 * 1000.0 / (2 * 3.0 * (1 + 1 / 4)))
    @test all(iszero, Array(dt₀)[[1, end], :])

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

    stokes.U.Ux .= 12.0
    JR2.displacement2velocity!(stokes, 4, DisplacementBoundaryConditions())
    @test all(stokes.V.Vx .== 3.0)

    velocity_before = copy(stokes.V.Vx)
    @test isnothing(
        JR2.displacement2velocity!(stokes, 4, VelocityBoundaryConditions())
    )
    @test stokes.V.Vx == velocity_before
    @test_throws "Unknown boundary conditions type: Nothing" JR2.displacement2velocity!(
        stokes, 4, nothing
    )
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

    # a fully populated, non-trivial symmetric stress tensor exercises the Jacobi
    # rotation sweeps in eigen_symmetric_3x3
    stokes.τ.xx .= 1.0
    stokes.τ.yy .= 2.0
    stokes.τ.zz .= 3.0
    stokes.τ.xy_c .= 0.5
    stokes.τ.xz_c .= 0.25
    stokes.τ.yz_c .= 0.75
    @test JR3.compute_principal_stresses!(stokes, σ) === nothing
    # eigenvector component magnitudes equal absolute eigenvalues; their sum
    # equals the trace |λ₁|+|λ₂|+|λ₃| ≥ trace = 6 here (all eigenvalues > 0)
    λ1 = sqrt(sum(Array(σ.σ1)[i, 1, 1, 1]^2 for i in 1:3))
    λ2 = sqrt(sum(Array(σ.σ2)[i, 1, 1, 1]^2 for i in 1:3))
    λ3 = sqrt(sum(Array(σ.σ3)[i, 1, 1, 1]^2 for i in 1:3))
    @test isapprox(λ1 + λ2 + λ3, 6.0; atol = 1.0e-6)

    # Non-uniform edge shear components: each is the mean of squares over the
    # four edges of its plane surrounding the cell center.
    εyz_e = [Float64(i + 2j + 3k) for i in 1:nx, j in 1:(ny + 1), k in 1:(nz + 1)]
    εxz_e = [Float64(2i - j + k) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
    εxy_e = [Float64(i * j + k) for i in 1:(nx + 1), j in 1:(ny + 1), k in 1:nz]
    stokes.ε_pl.xx .= 2.0
    stokes.ε_pl.yy .= -1.0
    stokes.ε_pl.zz .= -1.0
    copyto!(stokes.ε_pl.yz, εyz_e)
    copyto!(stokes.ε_pl.xz, εxz_e)
    copyto!(stokes.ε_pl.xy, εxy_e)
    εII_ref = [
        sqrt(
                0.5 * (4.0 + 1.0 + 1.0) +
                sum(abs2, εyz_e[i, j:(j + 1), k:(k + 1)]) / 4 +
                sum(abs2, εxz_e[i:(i + 1), j, k:(k + 1)]) / 4 +
                sum(abs2, εxy_e[i:(i + 1), j:(j + 1), k]) / 4
            ) for i in 1:nx, j in 1:ny, k in 1:nz
    ]
    JR3.accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, 0.5)
    @test Array(stokes.EII_pl) ≈ 0.5 .* εII_ref

    JR3.tensor_invariant!(stokes.ε_pl)
    @test Array(stokes.ε_pl.II) ≈ εII_ref

    stokes.EVol_pl .= -1.0
    stokes.ε_vol_pl .= 0.75
    JR3.accumulate_vol!(stokes.EVol_pl, stokes.ε_vol_pl, 4.0)
    @test all(Array(stokes.EVol_pl) .≈ 2.0)

    thermal = JR3.ThermalArrays(backend, ni)
    @test size(thermal.T) == (nx + 2, ny + 2, nz + 2)
    @test size(@view(thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)])) == ni
    @test parent(@view(thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)])) === thermal.T
    @test size(thermal.Told) == (nx + 2, ny + 2, nz + 2)
    @test size(thermal.ΔT) == (nx + 2, ny + 2, nz + 2)
    @test size(thermal.adiabatic) == ni
    @test size(thermal.dT_dt) == ni
    @test size(thermal.qTx) == (nx + 1, ny, nz)
    @test size(thermal.qTy) == (nx, ny + 1, nz)
    @test size(thermal.qTz) == (nx, ny, nz + 1)
    @test size(thermal.qTx2) == (nx + 1, ny, nz)
    @test size(thermal.qTy2) == (nx, ny + 1, nz)
    @test size(thermal.qTz2) == (nx, ny, nz + 1)
    @test size(thermal.ResT) == ni

    @test typeof(thermal.T) <: BackendArray
    @test typeof(thermal.Told) <: BackendArray
    @test typeof(thermal.ΔT) <: BackendArray
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

@testset "Type constructor: integer-only validation" begin
    @test_throws ArgumentError JustRelax.Velocity(10.0, 10.0)
    @test_throws ArgumentError JustRelax.Velocity(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.Displacement(10.0, 10.0)
    @test_throws ArgumentError JustRelax.Displacement(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.Vorticity((10.0, 10.0))
    @test_throws ArgumentError JustRelax.Vorticity((10.0, 10.0, 10.0))
    @test_throws ArgumentError JustRelax.Viscosity((10.0, 10.0))
    @test_throws ArgumentError JustRelax.Viscosity((10.0, 10.0, 10.0))
    @test_throws ArgumentError JustRelax.SymmetricTensor(10.0, 10.0)
    @test_throws ArgumentError JustRelax.SymmetricTensor(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.Residual(10.0, 10.0)
    @test_throws ArgumentError JustRelax.Residual(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.ThermalArrays(10.0, 10.0)
    @test_throws ArgumentError JustRelax.ThermalArrays(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.StokesArrays(10.0, 10.0)
    @test_throws ArgumentError JustRelax.StokesArrays(10.0, 10.0, 10.0)
    @test_throws ArgumentError JustRelax.RockRatio(10.0, 10.0)
    @test_throws ArgumentError JustRelax.RockRatio(10.0, 10.0, 10.0)
end
