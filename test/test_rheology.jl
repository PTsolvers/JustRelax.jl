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

using JustPIC, JustPIC._2D
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
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

        # tuple + phase_ratios overloads: dispatch through `fn_ratio(compute_density, …)`.
        # 70/30 mix of ρ=2700 and ρ=3300 ⇒ ρ_avg = 2880
        mat2 = GeoParams.SetMaterialParams(;
            Phase = 2,
            Density = GeoParams.ConstantDensity(; ρ = 3300.0),
            Gravity = GeoParams.ConstantGravity(; g = 9.81),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = 1.0e21),)),
        )
        rheology = (mat, mat2)
        ratios = SA[0.7, 0.3]
        ρ_avg = 0.7 * 2700.0 + 0.3 * 3300.0
        ρg_avg = ρ_avg * 9.81

        @test JustRelax2D.compute_buoyancy(rheology, args, ratios) ≈ ρg_avg
        b2 = JustRelax2D.compute_buoyancies(rheology, ratios, args, gvec, Val(2))
        @test b2[1] ≈ 0.0 && b2[2] ≈ -ρg_avg
        b3 = JustRelax2D.compute_buoyancies(rheology, ratios, args, gvec, Val(3))
        @test length(b3) == 3 && b3[3] ≈ -ρg_avg
        @test JustRelax2D.compute_buoyancies(rheology, ratios, args, -9.81, Val(2)) ≈ -ρg_avg
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

    @testset "Viscosity helpers" begin
        # src/rheology/Viscosity.jl: small pure helpers.

        # correct_phase_ratio — 3 explicit branches
        # 1) air_phase == 0 → passthrough
        r = SA[0.6, 0.4]
        @test JustRelax2D.correct_phase_ratio(0, r) === r

        # 2) ratio[air_phase] ≈ 1 → all-air → zeros
        @test JustRelax2D.correct_phase_ratio(1, SA[1.0, 0.0]) == SA[0.0, 0.0]

        # 3) general case → mask air slot and renormalize the rest
        # input [0.6, 0.4] with air=2 → mask → [0.6, 0.0] → renorm → [1.0, 0.0]
        @test JustRelax2D.correct_phase_ratio(2, SA[0.6, 0.4]) == SA[1.0, 0.0]
        # 60/40 rock split with air=3 leaves the rock untouched after renorm
        @test JustRelax2D.correct_phase_ratio(3, SA[0.3, 0.2, 0.5]) ≈ SA[0.6, 0.4, 0.0]

        # local_viscosity_args / local_args — extract value at index I and inject (dt=Inf, τII_old=0)
        T = reshape(collect(1.0:30.0), 3+2, 4+2)
        P = reshape(collect(101.0:112.0), 3, 4)
        args = (; T = T, P = P)
        la = JustRelax2D.local_viscosity_args(args, 2, 3)
        @test la.T == T[3, 4]
        @test la.P == P[2, 3]
        @test la.dt === Inf
        @test la.τII_old === 0.0
        la2 = JustRelax2D.local_args(args, 2, 3)
        @test la2.T == T[2, 3]

        # local_viscosity_args_vertex (2D): averages four cell-center neighbors
        la_v = JustRelax2D.local_viscosity_args_vertex(args, 2, 3)
        # Clamped indices: il=max(1,1)=1, ir=min(2,3)=2, jb=max(2,1)=2, jt=min(3,4)=3
        expected_T = 0.25 * (T[(1, 2).+1...] + T[(2, 2).+1...] + T[(1, 3).+1...] + T[(2, 3).+1...])
        @test la_v.T ≈ expected_T
        @test la_v.dt === Inf

        # local_viscosity_args_vertex (3D): averages eight cell-center neighbors
        T3 = reshape(collect(1.0:125.0), (3, 3, 3).+2...)
        P3 = reshape(collect(101.0:127.0), 3, 3, 3)
        args3 = (; T = T3, P = P3)
        la3 = JustRelax2D.local_viscosity_args_vertex(args3, 2, 2, 2)
        # il,ir,jb,jt,kf,kb all valid → sum of T3[1..2, 1..2, 1..2] / 8
        expected_T3 = sum(T3[i, j, k] for i in 2:3, j in 2:3, k in 2:3) / 8
        @test la3.T ≈ expected_T3

        # compute_phase_viscosity — single-phase dominance (ratio[i] > 0.999) → early-exit
        η1, η2 = 1.0e21, 1.0e23
        mat_v1 = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = GeoParams.ConstantDensity(; ρ = 2700.0),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = η1),)),
        )
        mat_v2 = GeoParams.SetMaterialParams(;
            Phase = 2,
            Density = GeoParams.ConstantDensity(; ρ = 3000.0),
            CompositeRheology = GeoParams.CompositeRheology((GeoParams.LinearViscous(; η = η2),)),
        )
        rheo2 = (mat_v1, mat_v2)
        v_args = (; T = 0.0, P = 0.0, dt = Inf, τII_old = 0.0)
        AII = 1.0e-15

        # Single-phase early exit: ratio[1] > 0.999 → returns η of phase 1
        η_dom = JustRelax2D.compute_phase_viscosity(rheo2, SA[1.0, 0.0], AII, GeoParams.compute_viscosity_εII, v_args)
        @test η_dom ≈ η1

        # Harmonic-mean path: 50/50 mix → 1 / (0.5/η1 + 0.5/η2)
        η_mix = JustRelax2D.compute_phase_viscosity(rheo2, SA[0.5, 0.5], AII, GeoParams.compute_viscosity_εII, v_args)
        @test η_mix ≈ inv(0.5 / η1 + 0.5 / η2)

        # Zero ratios are skipped in the harmonic sum (only mat_v2 contributes)
        η_only2 = JustRelax2D.compute_phase_viscosity(rheo2, SA[0.0, 0.8], AII, GeoParams.compute_viscosity_εII, v_args)
        @test η_only2 ≈ inv(0.8 / η2)

        # get_viscosity_fn — dispatcher
        @test JustRelax2D.get_viscosity_fn(GeoParams.compute_viscosity_εII) === JustRelax2D.compute_viscosity_εII!
        @test JustRelax2D.get_viscosity_fn(GeoParams.compute_viscosity_τII) === JustRelax2D.compute_viscosity_τII!
    end

    @testset "StressUpdate weighted paths" begin
        # src/rheology/StressUpdate.jl: the per-element/per-ratio walkers that
        # blend a yield function (and its gradients) across phases.

        # Tiny helper: muladd over an NTuple
        @test JustRelax2D._muladd_ntuple(0.5, (1.0, 2.0, 3.0), (10.0, 20.0, 30.0)) ===
            (10.5, 21.0, 31.5)

        # _zero_plastic_grad → tuple-of-zeros, dQdP=0, dFdP=0
        z_dQdτ, z_dQdP, z_dFdP = JustRelax2D._zero_plastic_grad((1.0, 2.0, 3.0), 5.0)
        @test z_dQdτ === (0.0, 0.0, 0.0)
        @test z_dQdP === 0.0
        @test z_dFdP === 0.0

        # Build a 2-phase rheology with one plastic and one purely elastic phase.
        pl = GeoParams.DruckerPrager_regularised(; C = 1.0, ϕ = 30.0, η_vp = 1.0e-3, Ψ = 0.0)
        elastic = GeoParams.ConstantElasticity(; G = 1.0, Kb = 1.0)
        visc = GeoParams.LinearViscous(; η = 1.0)
        mat_pl = GeoParams.SetMaterialParams(;
            Phase = 1,
            Density = GeoParams.ConstantDensity(; ρ = 0.0),
            CompositeRheology = GeoParams.CompositeRheology((visc, elastic, pl)),
        )
        mat_el = GeoParams.SetMaterialParams(;
            Phase = 2,
            Density = GeoParams.ConstantDensity(; ρ = 0.0),
            CompositeRheology = GeoParams.CompositeRheology((visc, elastic)),
        )
        rheology = (mat_pl, mat_el)
        args = (; P = 0.0, τII = 5.0, EII = 0.0)

        # Weighted F: with all weight on the plastic phase, equals the single-phase F.
        F_single = JustRelax2D.compute_yieldfunction_phase(rheology, 1; args...)
        F_w_plastic = JustRelax2D.compute_yieldfunction_phase(rheology, (1.0, 0.0); args...)
        @test F_w_plastic ≈ F_single

        # With all weight on the elastic-only phase, _yieldfunction_elements falls
        # through to `get(args, :τII, 0.0)` — so F = τII (5.0).
        F_w_elastic = JustRelax2D.compute_yieldfunction_phase(rheology, (0.0, 1.0); args...)
        @test F_w_elastic == 5.0

        # 50/50 blend = 0.5*F_plastic + 0.5*τII  (linear in the weights)
        F_mix = JustRelax2D.compute_yieldfunction_phase(rheology, (0.5, 0.5); args...)
        @test F_mix ≈ 0.5 * F_single + 0.5 * 5.0

        # SVector ratio path
        F_sv = JustRelax2D.compute_yieldfunction_phase(rheology, SA[0.5, 0.5]; args...)
        @test F_sv ≈ F_mix

        # Zero-weight phases are skipped (no contribution; doesn't NaN even though
        # the elastic phase would otherwise be evaluated)
        F_zero_skip = JustRelax2D.compute_yieldfunction_phase(rheology, (1.0, 0.0); args...)
        @test F_zero_skip ≈ F_single

        # Plastic gradients via the NTuple-ratio path
        τij = (1.0, -1.0, 0.5)
        dQdτ_t, dQdP_t, dFdP_t = JustRelax2D.compute_plastic_gradients_phase(
            rheology, (1.0, 0.0), τij; args...,
        )
        @test length(dQdτ_t) == 3
        # All weight on the elastic-only phase → _plastic_grad_elements never finds
        # a plastic primitive and returns _zero_plastic_grad → (0,0,0,0,0,0)
        dQdτ_e, dQdP_e, dFdP_e = JustRelax2D.compute_plastic_gradients_phase(
            rheology, (0.0, 1.0), τij; args...,
        )
        @test all(dQdτ_e .== 0.0)
        @test dQdP_e == 0.0
        @test dFdP_e == 0.0
    end

    @testset "PhaseRatios" begin
        # src/phases/PhaseRatios.jl: `update_phase_ratios_2D!` and its kernels.
        # `compute_dx` needs `LinRange` (or similar AbstractRange) xci/xvi —
        # passing plain Vector{Float64} causes `T = Vector{Float64}` to leak into
        # `@MVector zeros(T, N)`, which is a separate API constraint not relevant here.
        nx, ny = 4, 4
        pr = JustPIC._2D.PhaseRatios(backend_JP, 2, (nx, ny))
        xvi = (range(0.0, 1.0; length = nx + 1), range(0.0, 1.0; length = ny + 1))
        xci = (range(0.125, 0.875; length = nx), range(0.125, 0.875; length = ny))

        # Two-phase split: phase 1 on the left half, phase 2 on the right half.
        # Use ParallelStencil's architecture-agnostic allocators so the test runs
        # against the active backend's array type (CPU/CUDA/AMDGPU).
        p1 = @zeros(nx, ny)
        p2 = @zeros(nx, ny)
        p1[1:2, :] .= 1.0
        p2[3:4, :] .= 1.0
        phase_arrays = (p1, p2)

        JustRelax2D.update_phase_ratios_2D!(pr, phase_arrays, xci, xvi)

        # CellArrays stores SVector per cell; copy to host first so we can
        # safely index per-cell on any backend.
        center_h = Base.Array(pr.center)
        vertex_h = Base.Array(pr.vertex)
        Vx_h = Base.Array(pr.Vx)
        Vy_h = Base.Array(pr.Vy)

        # Cell-center ratios reproduce the input cleanly (only one phase per cell).
        @test center_h[1, 1][1] ≈ 1.0 && center_h[1, 1][2] ≈ 0.0
        @test center_h[2, 1][1] ≈ 1.0 && center_h[2, 1][2] ≈ 0.0
        @test center_h[3, 1][1] ≈ 0.0 && center_h[3, 1][2] ≈ 1.0
        @test center_h[4, 1][1] ≈ 0.0 && center_h[4, 1][2] ≈ 1.0

        # Every center cell should still sum to one
        @test all(sum(center_h[i, j]) ≈ 1.0 for i in 1:nx, j in 1:ny)

        # Vertex ratios sum to one as well
        @test all(sum(vertex_h[i, j]) ≈ 1.0 for i in 1:(nx + 1), j in 1:(ny + 1))

        # At a vertex on the phase boundary (i = 3), both phases contribute
        v = vertex_h[3, 2]
        @test v[1] > 0.0 && v[2] > 0.0

        # Vx- and Vy-face ratios sum to one
        @test all(sum(Vx_h[i, j]) ≈ 1.0 for i in axes(Vx_h, 1), j in axes(Vx_h, 2))
        @test all(sum(Vy_h[i, j]) ≈ 1.0 for i in axes(Vy_h, 1), j in axes(Vy_h, 2))

        # Threshold path: a tiny third phase (< 1e-5) should be cleaned to zero.
        pr3 = JustPIC._2D.PhaseRatios(backend_JP, 3, (nx, ny))
        p1b = @fill(0.6, nx, ny)
        p2b = @fill(0.4, nx, ny)
        p3b = @fill(1.0e-6, nx, ny)             # below the 1e-5 threshold
        JustRelax2D.update_phase_ratios_2D!(pr3, (p1b, p2b, p3b), xci, xvi)
        center3_h = Base.Array(pr3.center)
        @test center3_h[2, 2][3] == 0.0         # tiny phase zeroed out
        @test center3_h[2, 2][1] + center3_h[2, 2][2] ≈ 1.0
    end
end
