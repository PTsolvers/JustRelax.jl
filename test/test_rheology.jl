@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using StaticArrays
using GeoParams
using JustRelax, JustRelax.JustRelax2D

using ParallelStencil, ParallelStencil.FiniteDifferences2D
const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

@testset "Rheology" begin
    @testset "rheology/GeoParams helpers" begin
        # src/rheology/GeoParams.jl: get_α + bulk/shear modulus accessors over
        # the various GeoParams density / elasticity types.
        const_ρ = GeoParams.ConstantDensity(; ρ = 2700.0)
        T_ρ = GeoParams.T_Density(; ρ0 = 3000.0, T0 = 273.0, α = 3.0e-5)
        PT_ρ = GeoParams.PT_Density()
        meltX_ρ = GeoParams.Melt_DensityX()
        md = GeoParams.MeltDependent_Density(;
            ρsolid = GeoParams.T_Density(; α = 3.0e-5),
            ρmelt = GeoParams.T_Density(; α = 5.0e-5),
        )
        bf = GeoParams.BubbleFlow_Density(;
            ρmelt = GeoParams.T_Density(; α = 4.0e-5),
            ρgas = GeoParams.ConstantDensity(; ρ = 1.0),
            c0 = 0.01, a = 1.0,
        )
        gp = GeoParams.GasPyroclast_Density(;
            ρmelt = GeoParams.T_Density(; α = 4.0e-5),
            ρgas = GeoParams.ConstantDensity(; ρ = 1.0),
            δ = 0.5, β = 0.5,
        )

        # ConstantDensity branches → α = 0
        @test JustRelax2D.get_α(const_ρ) == 0
        @test JustRelax2D.get_α(const_ρ, (T = 300.0,)) == 0

        # T_Density / PT_Density / Melt_DensityX
        @test JustRelax2D.get_α(T_ρ) ≈ 3.0e-5
        @test JustRelax2D.get_α(T_ρ, (T = 300.0,)) ≈ 3.0e-5
        @test JustRelax2D.get_α(PT_ρ) ≈ 3.0e-5
        @test JustRelax2D.get_α(PT_ρ, (T = 300.0, P = 0.0)) ≈ 3.0e-5
        @test JustRelax2D.get_α(meltX_ρ) > 0
        @test JustRelax2D.get_α(meltX_ρ, (T = 300.0,)) > 0

        # MeltDependent_Density: pure-solid (ϕ=0) → αsolid, pure-melt (ϕ=1) → αmelt
        @test JustRelax2D.get_α(md; ϕ = 0.0) ≈ 3.0e-5
        @test JustRelax2D.get_α(md; ϕ = 1.0) ≈ 5.0e-5
        @test JustRelax2D.get_α(md, (T = 300.0,)) ≈ 3.0e-5     # strip-args overload

        # BubbleFlow_Density: only check it returns without erroring (the
        # formula uses 1/αgas of a ConstantDensity which is 0 ⇒ NaN, by design)
        @test JustRelax2D.get_α(bf; P = -1.0e5) isa Real
        @test JustRelax2D.get_α(bf, (P = 1.0e5,)) isa Real

        # GasPyroclast_Density
        @test JustRelax2D.get_α(gp) ≈ 0.5 * 4.0e-5     # δ·αgas + (1−δ)·αmelt with αgas=0
        @test JustRelax2D.get_α(gp, (P = 1.0e5,)) ≈ 0.5 * 4.0e-5

        # MaterialParams forwarding (single- and two-arg)
        elastic = GeoParams.ConstantElasticity(; G = 1.0e10, Kb = 5.0e10)
        mat_T = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = T_ρ,
            Elasticity = elastic,
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e21), elastic)),
        )
        @test JustRelax2D.get_α(mat_T) ≈ 3.0e-5
        @test JustRelax2D.get_α(mat_T, (T = 300.0,)) ≈ 3.0e-5
        @test JustRelax2D.get_thermal_expansion(mat_T) ≈ 3.0e-5

        # bulk/shear modulus: present → returns the value; absent → Inf fallback
        @test JustRelax2D.get_bulk_modulus(mat_T) ≈ 5.0e10
        @test JustRelax2D.get_shear_modulus(mat_T) ≈ 1.0e10
        mat_no_elastic = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = const_ρ,
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e21),)),
        )
        @test JustRelax2D.get_bulk_modulus(mat_no_elastic) === Inf
        @test JustRelax2D.get_shear_modulus(mat_no_elastic) === Inf
    end

    @testset "thermal rheology helpers" begin
        # exercise the compute_diffusivity / compute_ρCp / compute_α overloads
        # in src/thermal_diffusion/DiffusionPT_GeoParams.jl that are otherwise
        # only reached through full thermal-solver integration tests.
        mat = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = GeoParams.ConstantDensity(; ρ = 2700.0),
            HeatCapacity = GeoParams.ConstantHeatCapacity(; Cp = 1000.0),
            Conductivity = GeoParams.ConstantConductivity(; k = 3.0),
            Gravity = GeoParams.ConstantGravity(; g = 0.0),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e20),)),
        )
        rheology = (mat,)
        args = (; T = 300.0, P = 0.0)
        ratios = SA[1.0]

        κ_expected = 3.0 / (1000.0 * 2700.0)
        ρCp_expected = 2700.0 * 1000.0

        # no-phase signatures take a single MaterialParams
        @test JustRelax2D.compute_diffusivity(mat, args) ≈ κ_expected
        @test JustRelax2D.compute_ρCp(mat, args) ≈ ρCp_expected
        @test JustRelax2D.compute_diffusivity(mat, 2700.0, args) ≈ κ_expected
        @test JustRelax2D.compute_ρCp(mat, 2700.0, args) ≈ ρCp_expected

        # phase-indexed signatures take a tuple of MaterialParams
        @test JustRelax2D.compute_diffusivity(rheology, 1, args) ≈ κ_expected
        @test JustRelax2D.compute_diffusivity(rheology, 2700.0, 1, args) ≈ κ_expected
        @test JustRelax2D.compute_ρCp(rheology, 1, args) ≈ ρCp_expected
        @test JustRelax2D.compute_ρCp(rheology, 2700.0, 1, args) ≈ ρCp_expected

        # phase-ratio signatures
        @test JustRelax2D.compute_diffusivity(rheology, ratios, args) ≈ κ_expected
        @test JustRelax2D.compute_ρCp(rheology, ratios, args) ≈ ρCp_expected
        @test JustRelax2D.compute_ρCp(rheology, 2700.0, ratios, args) ≈ ρCp_expected

        @test JustRelax2D.compute_α(rheology, ratios) == 0.0
    end

    @testset "BuoyancyForces helpers" begin
        # src/rheology/BuoyancyForces.jl: pure compute_buoyancy / compute_buoyancies
        # overloads + the trait-dispatched update_ρg! no-op for constant density.
        mat = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = GeoParams.ConstantDensity(; ρ = 2700.0),
            Gravity = GeoParams.ConstantGravity(; g = 9.81),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e21),)),
        )
        args = (; T = 300.0, P = 0.0)

        ρg_expected = 2700.0 * 9.81

        # compute_buoyancy(::MaterialParams, args) — ρ·g
        @test JustRelax2D.compute_buoyancy(mat, args) ≈ ρg_expected

        # compute_buoyancies: 2D (planar) keeps gᵢ[1] and gᵢ[3]
        gvec = (0.0, 0.0, -9.81)
        b2 = JustRelax2D.compute_buoyancies(mat, args, gvec, Val(2))
        @test length(b2) == 2
        @test b2[1] ≈ 0.0 && b2[2] ≈ -ρg_expected

        # compute_buoyancies: 3D uses the whole gravity vector
        b3 = JustRelax2D.compute_buoyancies(mat, args, gvec, Val(3))
        @test length(b3) == 3
        @test b3[3] ≈ -ρg_expected

        # compute_buoyancies: scalar gravity → ρ·g scalar
        @test JustRelax2D.compute_buoyancies(mat, args, -9.81, Val(2)) ≈ -ρg_expected

        # fill_density! tuple variant writes each component
        ρg_tuple = (zeros(2, 2), zeros(2, 2))
        JustRelax2D.fill_density!(ρg_tuple, (1.0, 2.0), 1, 1)
        @test ρg_tuple[1][1, 1] == 1.0
        @test ρg_tuple[2][1, 1] == 2.0

        # fill_density! scalar variant only writes the last array (gravity along last axis)
        ρg_tuple2 = (zeros(2, 2), zeros(2, 2))
        JustRelax2D.fill_density!(ρg_tuple2, 3.0, 2, 2)
        @test ρg_tuple2[1] == zeros(2, 2)
        @test ρg_tuple2[2][2, 2] == 3.0

        # update_ρg!: ConstantDensityTrait → no-op (array stays untouched)
        ρg = ones(2, 2)
        JustRelax2D.update_ρg!(ρg, (mat,), args)
        @test all(ρg .== 1.0)

    end

    @testset "Melting.jl" begin
        # src/rheology/Melting.jl: the no-phase-ratios kernel call path is the
        # one not covered by existing integration tests (Volcano2D / thermalstresses
        # go through the JustPIC.PhaseRatios overload).
        mat = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = GeoParams.ConstantDensity(; ρ = 2700.0),
            Gravity = GeoParams.ConstantGravity(; g = 9.81),
            Melting = GeoParams.MeltingParam_Caricchi(),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e21),)),
        )

        ni = (4, 4)
        ϕ = zeros(ni...)
        T_hot = fill(1373.15, ni...)        # ≈ 1100 °C, well above the solidus
        T_cold = fill(573.15, ni...)        # ≈ 300 °C, well below the solidus
        P = fill(1.0e8, ni...)

        compute_melt_fraction!(ϕ, mat, (T = T_hot, P = P))
        @test all(0.95 .< ϕ .< 1.0)         # nearly fully molten

        ϕ .= 0.0
        compute_melt_fraction!(ϕ, mat, (T = T_cold, P = P))
        # subsolidus: MeltingParam_Caricchi returns ~3e-10 at 300°C, not exactly 0
        @test all(ϕ .< 1.0e-6)
    end
end
