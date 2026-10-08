@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(false)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
    CUDA.allowscalar(false)
end

using Test, GeoParams, JustRelax

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

@testset "Single-material shear heating $N D" for (N, JR) in (
        (2, JustRelax.JustRelax2D), (3, JustRelax.JustRelax3D),
    )
    ni = ntuple(_ -> 4, N)
    stokes = JR.StokesArrays(backend, ni)
    thermal = JR.ThermalArrays(backend, ni)
    elasticity = ConstantElasticity(; G = 2.0, Kb = 4.0)
    rheology = SetMaterialParams(;
        Phase = 1,
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
        ShearHeat = ConstantShearheating(1.0NoUnits),
    )
    stokes.τ.xx .= 2.0
    stokes.τ.yy .= -2.0
    stokes.ε.xx .= 3.0
    stokes.ε.yy .= -3.0

    # Elastic strain rate is ±0.5; heating is 2 * 2 * (3 - 0.5) = 10.
    JR.compute_shear_heating!(thermal, stokes, rheology, 1.0)
    @test all(==(10.0), Array(thermal.shear_heating))

    stokes.ε.xx .= -3.0
    stokes.ε.yy .= 3.0
    JR.compute_shear_heating!(thermal, stokes, rheology, 1.0)
    @test all(iszero, Array(thermal.shear_heating))

    # `compute_viscosity!` has no `(η, ν, εII, args, rheology, cutoff)` method on any backend.
    @test !applicable(JR.compute_viscosity!, stokes.viscosity.η, 1.0, stokes.ε.II, (;), rheology, (0.0, Inf))
end
