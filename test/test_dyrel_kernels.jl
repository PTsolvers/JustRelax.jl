push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
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

        stokes = StokesArrays(backend, ni)
        a, b = 2.0, -0.7
        stokes.V.Vx .= PTArray(backend)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        JR2K.compute_∇V_strain_rate!(stokes, _di, ni, Val(2))

        div = a + b
        @test all(Array(stokes.∇V) .≈ div)
        @test all(Array(stokes.ε.xx) .≈ a - div / 3)
        @test all(Array(stokes.ε.yy) .≈ b - div / 3)
        @test all(abs.(Array(stokes.ε.xy)) .< 1.0e-12)
    end

    @testset "standard APT free-surface pseudo-time" begin
        nx, ny = 5, 4
        ni = nx, ny
        grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
        stokes = StokesArrays(backend, ni)
        stokes.V.Vy .= 1.0
        ρgx = @zeros(ni...)
        ρgy = PTArray(backend)([3.0 - y for _ in 1:nx, y in 1:ny])
        ητ = @ones(ni...) .* 2.0
        ηdτ = 0.5
        dt = 0.25
        i, j = 2, 2
        Vy_old = Array(stokes.V.Vy)[i + 1, j + 1]

        @parallel (@idx ni) JR2K.compute_V!(
            stokes.V.Vx, stokes.V.Vy, stokes.P,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
            ηdτ, ρgx, ρgy, ητ,
            grid._di.center, grid._di.vertex, dt,
        )

        _dy = grid._di.center[2]
        ρgy_host = Array(ρgy)
        ρg_S, ρg_N = ρgy_host[i, j], ρgy_host[i, j + 1]
        c_fs = JustRelax2D.free_surface_diagonal(ρg_S, ρg_N, _dy, dt)
        residual = -JustRelax2D._av_ya(ρgy_host, i, j) - c_fs * Vy_old
        expected = Vy_old + residual * JustRelax2D.free_surface_pseudotime(ηdτ, 2.0, c_fs)
        @test Array(stokes.V.Vy)[i + 1, j + 1] ≈ expected
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

        stokes = StokesArrays(backend, ni)
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

        # analytic pure-strain velocity field ⇒ known divergence a+b
        a, b = 1.3, -0.4
        stokes.V.Vx .= PTArray(backend)([a * x for x in xvi[1], _ in 1:(ny + 2)])
        stokes.V.Vy .= PTArray(backend)([b * y for _ in 1:(nx + 2), y in xvi[2]])

        dyrel = DYREL(backend, stokes, rheology, phase_ratios, di, dt; ϵ = 1.0e-6)

        # --- fused divergence + strain rate + pressure residual ---
        # P0 = P and Q = 0 ⇒ RP = -∇V = -(a+b), independent of ηb
        stokes.P0 .= stokes.P
        stokes.Q .= 0.0
        JR2K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt; args...)
        @test all(Array(stokes.R.RP) .≈ -(a + b))
        @test all(Array(stokes.ε.xx) .≈ a - (a + b) / 3)

        # --- fused stress + τII viscosity refresh (nonlinear branch) ---
        θc = copy(dyrel.P_num)
        η_before = copy(stokes.viscosity.η)
        JR2K.compute_stress_viscosity_DRYEL!(
            stokes, θc, dyrel.γ_eff, rheology, phase_ratios,
            1.0, dt, 1.0, args, (-Inf, Inf), false,
        )
        @test all(isfinite, Array(stokes.viscosity.η))
        @test all(>(0), Array(stokes.viscosity.η))
        @test all(isfinite, Array(stokes.viscosity.ηv))
        # θc assembles the small pressure correction γ_eff·RP + ΔPψ
        @test Array(θc) ≈ Array(dyrel.γ_eff) .* Array(stokes.R.RP) .+ Array(stokes.ΔPψ)

        # --- Powell-Hestenes velocity residual (no D division: safe) ---
        ρg = @zeros(ni...), @zeros(ni...)
        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
            _di.center, _di.vertex,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test all(isfinite, Array(stokes.R.Ry))

        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.V.Vx, stokes.V.Vy,
            stokes.P, stokes.ΔPψ,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
            _di.center, _di.vertex, 0.0,
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
            0.0,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test Array(stokes.V.Vx) == Array(Vx_before)
        @test Array(stokes.V.Vy) == Array(Vy_before)

        # The stabilized residual adds Vy*dt*∂y(ρg), using the same local
        # density-gradient coefficient included in the DYREL diagonal.
        ρg[2] .= PTArray(backend)([3.0 - y for _ in 1:nx, y in 1:ny])
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
            0.0,
        )
        Ry_without_fs = copy(stokes.R.Ry)
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
            0.5,
        )
        ∂ρg∂y = -inv(grid.di.center[2])
        expected_correction = Array(stokes.V.Vy[2:(end - 1), 2:(end - 1)]) .* (0.5 * ∂ρg∂y)
        @test Array(stokes.R.Ry) .- Array(Ry_without_fs) ≈ expected_correction

        # --- fused adjoint Schur correction + preconditioning + update ---
        adjoint = JustRelax2D.AdjointStokesArrays(backend, ni)
        adjoint.P .= 1.0
        adjoint.V.Vx .= 1.0
        adjoint.V.Vy .= 1.0
        adjoint.ε.xx .= 1.0
        adjoint.ε.yy .= 1.0
        adjoint.ε.xy .= 1.0
        adjoint.τ.xx .= 1.0
        adjoint.τ.yy .= 1.0
        adjoint.τ.xy .= 1.0
        adjoint.λV.Vx .= 2.0
        adjoint.λV.Vy .= 3.0
        adjoint.λP .= 4.0
        JR2K.initialize_adjoint_iteration!(adjoint, ni)
        @test all(iszero, Array(adjoint.P))
        @test all(iszero, Array(adjoint.V.Vx))
        @test all(iszero, Array(adjoint.V.Vy))
        @test all(iszero, Array(adjoint.ε.xx))
        @test all(iszero, Array(adjoint.ε.yy))
        @test all(iszero, Array(adjoint.ε.xy))
        @test all(iszero, Array(adjoint.τ.xx))
        @test all(iszero, Array(adjoint.τ.yy))
        @test all(iszero, Array(adjoint.τ.xy))
        @test all(==(2.0), Array(adjoint.R.Rx))
        @test all(==(3.0), Array(adjoint.R.Ry))
        @test all(==(4.0), Array(adjoint.R.RP))

        adjoint.V.Vx .= 4.0
        adjoint.V.Vy .= 5.0
        adjoint.P .= PTArray(backend)([i + 2j for i in 1:nx, j in 1:ny])
        adjoint.λV.Vx .= 0.0
        adjoint.λV.Vy .= 0.0
        dyrel.γ_eff .= 2.0
        dyrel.Dx .= 2.0
        dyrel.Dy .= 4.0
        dyrel.αVx .= 0.0
        dyrel.αVy .= 0.0
        dyrel.βVx .= 0.5
        dyrel.βVy .= 0.5
        dyrel.dτVx .= 0.25
        dyrel.dτVy .= 0.25
        dyrel.dVxdτ .= 0.0
        dyrel.dVydτ .= 0.0

        Rx0 = Array(adjoint.V.Vx)
        Ry0 = Array(adjoint.V.Vy)
        P = Array(adjoint.P)
        γ = Array(dyrel.γ_eff)
        expected_Rx = @views Rx0[2:(end - 1), 2:(end - 1)] .-
            (γ[1:(end - 1), :] .* P[1:(end - 1), :] .- γ[2:end, :] .* P[2:end, :]) .* _di.center[1]
        expected_Ry = @views Ry0[2:(end - 1), 2:(end - 1)] .-
            (γ[:, 1:(end - 1)] .* P[:, 1:(end - 1)] .- γ[:, 2:end] .* P[:, 2:end]) .* _di.center[2]

        @parallel (@idx ni) JR2K.update_adjoint_V_damping_DR!(
            adjoint.λV.Vx, adjoint.λV.Vy,
            dyrel.dVxdτ, dyrel.dVydτ,
            adjoint.V.Vx, adjoint.V.Vy,
            adjoint.P, dyrel.γ_eff,
            dyrel.Dx, dyrel.Dy,
            dyrel.αVx, dyrel.αVy,
            dyrel.βVx, dyrel.βVy,
            dyrel.dτVx, dyrel.dτVy,
            _di.center,
        )

        @test Array(adjoint.V.Vx[2:(end - 1), 2:(end - 1)]) ≈ expected_Rx
        @test Array(adjoint.V.Vy[2:(end - 1), 2:(end - 1)]) ≈ expected_Ry
        @test Array(dyrel.dVxdτ) ≈ expected_Rx ./ 2
        @test Array(dyrel.dVydτ) ≈ expected_Ry ./ 4
        @test Array(adjoint.λV.Vx[2:(end - 1), 2:(end - 1)]) ≈ expected_Rx ./ 16
        @test Array(adjoint.λV.Vy[2:(end - 1), 2:(end - 1)]) ≈ expected_Ry ./ 32

        # The adjoint penalty uses the local center spacing on a nonuniform grid.
        _di_nonuniform = (
            PTArray(backend)([1.0, 2.0, 3.0, 4.0, 5.0]),
            PTArray(backend)([2.0, 3.0, 4.0, 5.0]),
        )
        adjoint.V.Vx .= 4.0
        adjoint.V.Vy .= 5.0
        adjoint.λV.Vx .= 0.0
        adjoint.λV.Vy .= 0.0
        dyrel.dVxdτ .= 0.0
        dyrel.dVydτ .= 0.0
        expected_Rx = @views Rx0[2:(end - 1), 2:(end - 1)] .-
            (γ[1:(end - 1), :] .* P[1:(end - 1), :] .- γ[2:end, :] .* P[2:end, :]) .*
            reshape(Array(_di_nonuniform[1]), :, 1)
        expected_Ry = @views Ry0[2:(end - 1), 2:(end - 1)] .-
            (γ[:, 1:(end - 1)] .* P[:, 1:(end - 1)] .- γ[:, 2:end] .* P[:, 2:end]) .*
            reshape(Array(_di_nonuniform[2]), 1, :)

        @parallel (@idx ni) JR2K.update_adjoint_V_damping_DR!(
            adjoint.λV.Vx, adjoint.λV.Vy,
            dyrel.dVxdτ, dyrel.dVydτ,
            adjoint.V.Vx, adjoint.V.Vy,
            adjoint.P, dyrel.γ_eff,
            dyrel.Dx, dyrel.Dy,
            dyrel.αVx, dyrel.αVy,
            dyrel.βVx, dyrel.βVy,
            dyrel.dτVx, dyrel.dτVy,
            _di_nonuniform,
        )

        @test Array(adjoint.V.Vx[2:(end - 1), 2:(end - 1)]) ≈ expected_Rx
        @test Array(adjoint.V.Vy[2:(end - 1), 2:(end - 1)]) ≈ expected_Ry
    end
end

@testset "Adjoint backend allocation and pullback" begin
    ni = (5, 4)
    TA = PTArray(backend)
    stokes = StokesArrays(backend, ni)
    adjoint = AdjointStokesArrays(backend, ni, (true, false))
    @test adjoint.P isa TA
    @test adjoint.λV.Vx isa TA
    @test size(adjoint.R.Rx) == (ni[1], ni[2])
    @test AdjointStokesArrays(backend, ni...).ρ isa TA
    controls = material_controls(backend, ni, (:ρ0,); nphases = 2)
    @test controls.ρ0.center isa TA
    @test controls.ρ0.vertex isa TA
    @test size(controls.ρ0.center) == (2, ni...)
    @test size(controls.ρ0.vertex) == (2, (ni .+ 1)...)
    @test_throws ArgumentError material_controls(backend, ni, (:ρ0,))

    adjoint = AdjointStokesArrays(backend, ni)
    grid = Geometry(ni, (1.0, 1.0))
    observation = JR2K.observation_mask(
        adjoint, grid, (; field = :P, center = (0.5, 0.5), half_width = (0.5, 0.5))
    )
    JR2K.seed_observation!(observation)
    @test all(==(-1.0), Array(adjoint.P))

    # Exercise deferred Enzyme differentiation inside the selected device kernel.
    adjoint.R.Rx .= 1.0
    adjoint.R.Ry .= 1.0
    ρg = (TA(zeros(ni)), TA(ones(ni)))
    JR2K.enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, grid._di, ni)
    @test any(!iszero, Array(adjoint.τ.xy))
    @test any(!iszero, Array(adjoint.ρ))
    @test all(isfinite, Array(adjoint.P))
    expected_density = fill(-1.0, ni)
    expected_density[:, [1, end]] .= -0.5
    @test Array(adjoint.ρ) ≈ expected_density

    # With all rows in rock, the variational pullback agrees with the standard one.
    ϕ = RockRatio(backend, ni)
    for field in (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)
        fill!(field, 1.0)
    end
    masked_adjoint = AdjointStokesArrays(backend, ni)
    masked_adjoint.R.Rx .= 1.0
    masked_adjoint.R.Ry .= 1.0
    JR2K.enzyme_compute_PH_residual_V!(stokes, masked_adjoint, ρg, ϕ, grid._di, ni)
    @test Array(masked_adjoint.ρ) ≈ expected_density
    @test Array(masked_adjoint.τ.xy) ≈ Array(adjoint.τ.xy)
end


@testset "Continuity pressure pullback with infinite bulk viscosity" begin
    ni = (3, 2)
    stokes = StokesArrays(backend, ni)
    stokes.P .= 2.0
    phases = PhaseRatios(backend_JP, 1, ni)
    ϕ = RockRatio(backend, ni)
    foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
    grid = Geometry(ni, (1.0, 1.0))
    ηb = similar(stokes.P)
    for bulk in (5.0, Inf), seed in (1.0, 1.0e15), variational in (false, true)
        fill!(ηb, bulk)
        adjoint = AdjointStokesArrays(backend, ni)
        adjoint.R.RP .= seed
        if variational
            JR2K.enzyme_compute_∇V_strain_rate_RP!(
                stokes, adjoint, (; ηb), (), phases, ϕ, grid._di, ni, 3.0e11, (;)
            )
        else
            JR2K.enzyme_compute_∇V_strain_rate_RP!(
                stokes, adjoint, (; ηb), (), phases, grid._di, ni, 3.0e11, (;)
            )
        end
        @test Array(adjoint.P) ≈ fill(-seed / bulk, ni)
        @test all(isfinite, Array(adjoint.V.Vx))
        @test all(isfinite, Array(adjoint.V.Vy))
    end
end

@inline function material_gradient_test_point!(out, rheology, i, j)
    out[i, j] = rheology[1].a^2
    return nothing
end

@testset "Material-gradient annotations on the selected backend" begin
    ni = (3, 2)
    TA = PTArray(backend)
    out = TA(zeros(ni))
    seed = TA(ones(ni))
    fields = (; a = TA(zeros(1, ni...)))
    paths = ((; a = (Val((:a,)),)),)
    JR2K.differentiate_pointwise_material_kernel!(
        material_gradient_test_point!, ni, fields, paths,
        (out => seed, ((; a = 2.0),) => JR2K.Enzyme.Active),
    )
    @test Array(fields.a) == fill(4.0, 1, ni...)
    @test all(iszero, Array(seed))
end

@inline cap_test_λ(rheology, s, p) =
    JR2K.cap_return_mapping(rheology, 1, s, p, 0.0, 0.5, 4.0, 0.1)[1][3]

# Material derivative through the cap return map, whose Newton solve Enzyme skips. With an
# Active rheology handed to that solve, CUDA failed with an invalid address space (error 717).
@parallel_indices (i, j) function cap_rheology_gradient!(out, rheology, s, p)
    Enzyme = JR2K.Enzyme
    d = Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(cap_test_λ), Enzyme.Active,
        Enzyme.Active(rheology), Enzyme.Active(s), Enzyme.Active(p),
    )[1]
    out[1] = d[2]
    out[2] = d[3]
    out[3] = d[1][1].CompositeRheology[1].elements[3].C.val
    return nothing
end

@testset "Cap return-map material derivative on the selected backend" begin
    el = ConstantElasticity(; G = 1.0, Kb = 4.0)
    cap = DruckerPragerCap(; C = 1.0, ϕ = 30.0, Ψ = 0.0, η_vp = 0.1, pT = -0.5)
    rheology = (
        SetMaterialParams(;
            Phase = 1, Elasticity = el,
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, cap)),
        ),
    )
    s, p = 1.0, -0.8 # yields on the cap
    out = PTArray(backend)(zeros(3))
    @parallel (1:1, 1:1) cap_rheology_gradient!(out, rheology, s, p)
    Enzyme = JR2K.Enzyme
    host = Enzyme.autodiff(
        Enzyme.Reverse, Enzyme.Const(cap_test_λ), Enzyme.Active,
        Enzyme.Active(rheology), Enzyme.Active(s), Enzyme.Active(p),
    )[1]
    expected = [host[2], host[3], host[1][1].CompositeRheology[1].elements[3].C.val]
    @test !iszero(expected[3])
    @test Array(out) ≈ expected
end

@inline function periodic_neighbor_sum_point!(out, input, i, j)
    ip = i == size(out, 1) ? 1 : i + 1
    jp = j == size(out, 2) ? 1 : j + 1
    out[i, j] = input[i, j] + input[ip, j] + input[i, jp]
    return nothing
end

@testset "Race-free point coloring on the selected backend" begin
    TA = PTArray(backend)
    for ni in ((6, 4), (5, 3))
        input = TA(zeros(ni))
        input_ad = TA(zeros(ni))
        out = TA(zeros(ni))
        out_ad = TA(ones(ni))
        JR2K.enzyme_reverse_pointwise!(
            periodic_neighbor_sum_point!, ni, (out => out_ad, input => input_ad)
        )
        @test Array(input_ad) == fill(3.0, ni)
        @test all(iszero, Array(out_ad))
    end
end
