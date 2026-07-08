push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
using StaticArrays

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

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

@parallel_indices (i, j) function _init_single_phase!(phases)
    @index phases[1, i, j] = 1.0
    return nothing
end

@testset "DYREL kernels" begin

    # ------------------------------------------------------------------ #
    # Pure, analytic helpers (no grid / rheology needed)
    # ------------------------------------------------------------------ #
    @testset "pure helpers" begin
        # _compute_RP!(P, P0, ∇V, Q, ηb, dt) = -∇V - (P - P0)/ηb + Q/dt
        P, P0, ∇V, Q, ηb, dt = 3.0, 1.0, 0.5, 0.2, 4.0, 0.25
        expected = -∇V - (P - P0) / ηb + Q / dt
        @test JustRelax2D._compute_RP!(P, P0, ∇V, Q, ηb, dt) ≈ expected

        # thermal variant: _compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, dt)
        ΔT, α = 10.0, 3.0e-5
        expected_T = -∇V - (P - P0) / ηb + α * (ΔT / dt) + Q / dt
        @test JustRelax2D._compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, dt) ≈ expected_T

        # damped_update_V(dVdτ, R, α, β, dτ) = (dVdτ_new, dVdτ_new*β*dτ)
        dVdτ, R, a, b, dτ = 2.0, 0.5, 0.9, 0.8, 0.3
        dVdτ_new, ΔV = JustRelax2D.damped_update_V(dVdτ, R, a, b, dτ)
        @test dVdτ_new ≈ a * dVdτ + R
        @test ΔV ≈ (a * dVdτ + R) * b * dτ

        el = ConstantElasticity(; G = 1.0, Kb = 5.0)
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 0.0),
                Gravity = ConstantGravity(; g = 0.0),
                CompositeRheology = CompositeRheology(
                    (
                        LinearViscous(; η = 1.0),
                        el,
                        DruckerPrager_regularised(; C = 0.1, ϕ = 0.0, η_vp = 1.0e-2, Ψ = 0.0),
                    )
                ),
                Elasticity = el,
            ),
        )
        εij = (2.0, -1.0, 0.5)
        τij_o = (0.0, 0.0, 0.0)
        τ_plastic = JR2K.compute_local_stress(εij, τij_o, 1.0, 0.0, 0.0, 1.0, rheology, SA[1.0], 1.0, 0.0, true)
        τ_trial = JR2K.compute_local_stress(εij, τij_o, 1.0, 0.0, 0.0, 1.0, rheology, SA[1.0], 1.0, 0.0, false)
        @test τ_plastic[8] > 0.0
        @test τ_plastic[1] != τ_trial[1]
        @test JR2K.local_stress_dτxx_dεxx(εij, τij_o, 1.0, 0.0, 0.0, 1.0, rheology, SA[1.0], 1.0, 0.0) ≈ 1.0
    end

    # ------------------------------------------------------------------ #
    # Geometric divergence + deviatoric strain rate (2D), driven through
    # the public wrapper with an analytic pure-strain velocity field:
    #   Vx = a·x , Vy = b·y  ⇒  ∇V = a+b, εxx = a-(a+b)/3, εyy = b-(a+b)/3, εxy = 0
    # ------------------------------------------------------------------ #
    @testset "compute_∇V_strain_rate! 2D" begin
        nx, ny = 6, 5
        ni = nx, ny
        li = 1.0, 1.0
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        (; xvi) = grid
        _di = grid._di

        stokes = StokesArrays(backend_JR, ni)
        a, b = 2.0, -0.7
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        JR2K.compute_∇V_strain_rate!(stokes, _di, ni, Val(2))

        div = a + b
        @test all(Array(stokes.∇V) .≈ div)
        @test all(Array(stokes.ε.xx) .≈ a - div / 3)
        @test all(Array(stokes.ε.yy) .≈ b - div / 3)
        @test all(abs.(Array(stokes.ε.xy)) .< 1.0e-12)
    end

    # ------------------------------------------------------------------ #
    # Fused DYREL kernels (2D). A tiny single-phase, viscoelastic setup
    # drives the fused strain-rate+RP, stress+τII-viscosity (nonlinear
    # `linear_viscosity = false` branch), and the residual kernels.
    # ------------------------------------------------------------------ #
    @testset "fused kernels 2D" begin
        nx, ny = 6, 5
        ni = nx, ny
        li = 1.0, 1.0
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        (; xvi) = grid
        di = grid.di
        _di = grid._di
        dt = 1.0

        el = ConstantElasticity(; G = 1.0, Kb = 5.0)
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 1.0),
                Gravity = ConstantGravity(; g = 1.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el)),
                Elasticity = el,
            ),
        )

        phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
        @parallel (@idx ni) _init_single_phase!(phase_ratios.center)
        @parallel (@idx ni .+ 1) _init_single_phase!(phase_ratios.vertex)

        stokes = StokesArrays(backend_JR, ni)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

        # analytic pure-strain velocity field ⇒ known divergence a+b
        a, b = 1.3, -0.4
        stokes.V.Vx .= PTArray(backend_JR)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend_JR)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, di, dt; ϵ = 1.0e-6)

        # --- fused divergence + strain rate + pressure residual ---
        # P0 = P and Q = 0 ⇒ RP = -∇V = -(a+b), independent of ηb
        stokes.P0 .= stokes.P
        stokes.Q .= 0.0
        JR2K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args)
        @test all(Array(stokes.R.RP) .≈ -(a + b))
        @test all(Array(stokes.ε.xx) .≈ a - (a + b) / 3)

        # --- fused stress + τII viscosity refresh (nonlinear branch) ---
        θc = copy(dyrel.P_num)
        η_before = copy(stokes.viscosity.η)
        JR2K.compute_stress_viscosity_DRYEL!(
            stokes, θc, dyrel.γ_eff, rheology, phase_ratios,
            1.0, dt, 1.0, args, (-Inf, Inf), false, dyrel, true,
        )
        @test all(isfinite, Array(stokes.viscosity.η))
        @test all(>(0), Array(stokes.viscosity.η))
        @test all(isfinite, Array(stokes.viscosity.ηv))
        # θc assembles the small pressure correction γ_eff·RP + ΔPψ
        @test Array(θc) ≈ Array(dyrel.γ_eff) .* Array(stokes.R.RP) .+ Array(stokes.ΔPψ)
        @test dyrel.∂τxxc_∂εxx[2, 2] ≈ 1.0
        @test dyrel.∂τyyc_∂εyy[2, 2] ≈ 1.0
        @test dyrel.∂τxyv_∂εxy[2, 2] ≈ 1.0

        # --- Powell-Hestenes velocity residual (no D division: safe) ---
        ρg = @zeros(ni...), @zeros(ni...)
        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
            _di.center, _di.vertex,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test all(isfinite, Array(stokes.R.Ry))

        # --- fused DR residual + damped velocity update ---
        # D = 1, β = 0 ⇒ ΔV = 0 (velocity unchanged), residuals finite
        dyrel.Dx .= 1.0; dyrel.Dy .= 1.0
        dyrel.βVx .= 0.0; dyrel.βVy .= 0.0
        dyrel.αVx .= 0.0; dyrel.αVy .= 0.0
        dyrel.dτVx .= 1.0; dyrel.dτVy .= 1.0
        Vx_before = copy(stokes.V.Vx)
        Vy_before = copy(stokes.V.Vy)
        @parallel (@idx ni) JR2K.compute_DR_residual_update_V!(
            stokes.R.Rx, stokes.R.Ry,
            stokes.V.Vx, stokes.V.Vy,
            dyrel.dVxdτ, dyrel.dVydτ,
            stokes.P, θc,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
            ρg...,
            dyrel.Dx, dyrel.Dy,
            dyrel.αVx, dyrel.αVy,
            dyrel.βVx, dyrel.βVy,
            dyrel.dτVx, dyrel.dτVy,
            _di.center, _di.vertex,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test Array(stokes.V.Vx) == Array(Vx_before)
        @test Array(stokes.V.Vy) == Array(Vy_before)
    end
end
