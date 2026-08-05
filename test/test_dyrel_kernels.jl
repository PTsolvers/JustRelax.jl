push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, LinearAlgebra
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

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

# Assemble the reduced variational velocity Jacobian by applying the actual matrix-free DYREL
# kernels to unit velocity vectors. This is intentionally tiny: it catches mask/operator/
# Gershgorin disagreements without maintaining a second hand-written stencil.
function reduced_velocity_matrix(phi_case, ni = (4, 4); cut_fraction = 0.35)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    rheology = (SetMaterialParams(;
        Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
    ),)
    phase_ratios = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx ni) _init_single_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _init_single_phase!(phase_ratios.vertex)

    ϕ = RockRatio(backend_JR, ni)
    for a in (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)
        a .= 1
    end
    if phi_case == :cut
        cut_j = ni[2] ÷ 2 + 1
        ϕ.center[:, cut_j] .= cut_fraction
        ϕ.center[:, (cut_j + 1):end] .= 0
        ϕ.vertex[:, cut_j + 1] .= cut_fraction
        ϕ.vertex[:, (cut_j + 2):end] .= 0
        ϕ.Vx[:, cut_j] .= cut_fraction
        ϕ.Vx[:, (cut_j + 1):end] .= 0
        ϕ.Vy[:, cut_j + 1] .= cut_fraction
        ϕ.Vy[:, (cut_j + 2):end] .= 0
    end

    stokes = StokesArrays(backend_JR, ni)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = 1.0)
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, (-Inf, Inf); air_phase = 0)
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, ϕ, grid.di, 1.0; γfact = 20.0)
    gersh_D = Array(dyrel.Dx), Array(dyrel.Dy)
    gersh_bounds = Array(dyrel.λmaxVx), Array(dyrel.λmaxVy)
    ρg = @zeros(ni...), @zeros(ni...)
    θc = dyrel.P_num

    # The reduced space is whatever the kernels solve on, so the degrees of freedom have to be
    # enumerated with the solver's own face predicate: a face carrying rock is still eliminated
    # when its stencil reaches a void cell or vertex.
    maskV = (
        [JR2K.isvalid_vx_strict(ϕ, i + 1, j) for i in axes(stokes.R.Rx, 1), j in axes(stokes.R.Rx, 2)],
        [JR2K.isvalid_vy_strict(ϕ, i, j + 1) for i in axes(stokes.R.Ry, 1), j in axes(stokes.R.Ry, 2)],
    )
    dofs = [(d, I) for d in 1:2 for I in CartesianIndices(maskV[d]) if maskV[d][I]]
    A = zeros(length(dofs), length(dofs))

    for column in eachindex(dofs)
        stokes.V.Vx .= 0
        stokes.V.Vy .= 0
        d, I = dofs[column]
        (d == 1 ? stokes.V.Vx : stokes.V.Vy)[I[1] + 1, I[2] + 1] = 1
        stokes.P .= 0
        stokes.P0 .= 0
        stokes.Q .= 0
        stokes.ΔPψ .= 0
        stokes.λ .= 0
        stokes.λv .= 0
        JR2K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, grid._di, ni, 1.0, args, true)
        JR2K.compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, 1.0, 1.0)
        @. θc = dyrel.γ_eff * stokes.R.RP
        dyrel.Dx .= 1
        dyrel.Dy .= 1
        dyrel.αVx .= 0
        dyrel.αVy .= 0
        dyrel.βVx .= 0
        dyrel.βVy .= 0
        dyrel.dτVx .= 0
        dyrel.dτVy .= 0
        @parallel (@idx ni) JR2K.compute_DR_residual_update_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.V.Vx, stokes.V.Vy,
            dyrel.dVxdτ, dyrel.dVydτ, stokes.P, θc, stokes.τ.xx, stokes.τ.yy,
            stokes.τ.xy, ρg..., dyrel.Dx, dyrel.Dy, dyrel.αVx, dyrel.αVy,
            dyrel.βVx, dyrel.βVy, dyrel.dτVx, dyrel.dτVy, ϕ,
            grid._di.center, grid._di.vertex, 0.0,
        )
        R = Array(stokes.R.Rx), Array(stokes.R.Ry)
        A[:, column] .= [-R[d][I] for (d, I) in dofs]
    end
    return A, dofs, gersh_D, gersh_bounds
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
            _di.center, _di.vertex, 0.0,
        )
        @test all(isfinite, Array(stokes.R.Rx))
        @test Array(stokes.V.Vx) == Array(Vx_before)
        @test Array(stokes.V.Vy) == Array(Vy_before)
    end

    @testset "masked stencil ϕ=0 guard" begin
        # A fully-masked neighbor (ϕ == 0) must contribute exactly 0 even when its stored value is
        # non-finite — air cells hold τ = Inf under an unbounded viscosity cutoff — so `Inf * 0 =
        # NaN` cannot leak into a valid cell's residual stencil. A non-masked neighbor (ϕ != 0)
        # still propagates a non-finite value, preserving fail-fast for genuine interior NaNs.
        A = fill(Inf, 3, 3)
        ϕ0 = zeros(3, 3)
        ϕ1 = ones(3, 3)
        for f in (JR2K.center, JR2K.front, JR2K.right, JR2K.next)
            @test f(A, ϕ0, 1, 1) == 0.0
            @test !isfinite(f(A, ϕ1, 1, 1))
        end
        # a partially-valid neighbor (ϕ != 0) at a non-finite cell must NOT be silenced
        Ap = ones(3, 3); Ap[1, 1] = Inf
        @test !isfinite(JR2K.center(Ap, fill(0.5, 3, 3), 1, 1))
    end

    @testset "reduced pressure mask" begin
        if backend_JR === CPUBackend
            ϕ = RockRatio(backend_JR, (3, 3))
            for A in (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)
                A .= 1
            end
            # The center remains geometrically non-empty, but its north velocity face has zero
            # weight, so its divergence constraint must be eliminated from the reduced system.
            ϕ.Vy[2, 3] = 0
            maskP = ϕ.center .> 0
            @parallel (@idx size(maskP)) JR2K.update_valid_c_mask!(maskP, ϕ)
            @test !Array(maskP)[2, 2]
            @test Array(maskP)[1, 1]
        end
    end

    # ------------------------------------------------------------------ #
    # FSSA (free-surface stabilization) helpers added to the DYREL
    # preconditioner/penalty pipeline (Gershgorin.jl, constructors.jl,
    # solver.jl, velocity_kernels_VS.jl).
    # ------------------------------------------------------------------ #
    @testset "fssa_diagonal_y" begin
        ρgy = reshape(collect(1.0:12.0), 3, 4)  # nx=3, ny=4
        _dy = 2.0
        dt = 0.5
        i, j = 2, 2

        # no buoyancy field ⇒ exact zero, independent of everything else
        @test JR2K.fssa_diagonal_y(nothing, i, j, _dy, dt) == false

        # last row: no north neighbour ⇒ exact zero
        @test JR2K.fssa_diagonal_y(ρgy, i, size(ρgy, 2), _dy, dt) == 0.0

        # interior, unmasked: matches the plain finite difference
        expected = JR2K._d_ya(ρgy, _dy, i, j) * dt
        @test JR2K.fssa_diagonal_y(ρgy, i, j, _dy, dt) ≈ expected

        # interior, ϕ-masked: matches the masked finite difference through ϕ.center. Host arrays
        # throughout: `fssa_diagonal_y` is called directly, outside any kernel.
        ϕ = JustRelax.JustRelax2D.RockRatio(CPUBackend, (3, 4))
        ϕ.center .= reshape(collect(0.1:0.1:1.2), 3, 4)
        expected_ϕ = JR2K._d_ya(ρgy, ϕ.center, _dy, i, j) * dt
        @test JR2K.fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ) ≈ expected_ϕ
    end

    @testset "set_preconditioner!" begin
        D = zeros(2, 2)
        λmax = zeros(2, 2)

        JR2K.set_preconditioner!(D, λmax, 4.0, 8.0, 1, 1)
        @test D[1, 1] == 4.0
        @test λmax[1, 1] == 2.0  # inv(4.0) * 8.0

        # non-positive diagonal (degenerate mask-boundary cell) falls back to 1/1,
        # matching the pre-existing invalid-cell branch
        JR2K.set_preconditioner!(D, λmax, -1.0, 8.0, 2, 1)
        @test D[2, 1] == 1.0 && λmax[2, 1] == 1.0
        JR2K.set_preconditioner!(D, λmax, 0.0, 8.0, 1, 2)
        @test D[1, 2] == 1.0 && λmax[1, 2] == 1.0
    end

    @testset "ϕ_weighted_harmonic" begin
        # G·dt > 0: viscoelastic harmonic combine, as before the fix
        @test JR2K.ϕ_weighted_harmonic(1.0, 2.0, 4.0, 0.5) ≈ 1.0

        # ϕ == 0 ⇒ exact zero regardless of η, G, dt
        @test JR2K.ϕ_weighted_harmonic(0.0, 2.0, 4.0, 0.5) == 0.0

        # G == Inf (no elasticity) ⇒ same elastic-free combine ϕ·η as before the fix
        @test JR2K.ϕ_weighted_harmonic(1.0, 2.0, Inf, 0.5) ≈ 2.0

        # G == 0 or NaN (degenerate modulus): falls back to ϕ·η instead of a naive
        # inv(G*dt) poisoning the row with Inf/NaN
        @test JR2K.ϕ_weighted_harmonic(1.0, 2.0, 0.0, 0.5) ≈ 2.0
        @test JR2K.ϕ_weighted_harmonic(1.0, 2.0, NaN, 0.5) ≈ 2.0
    end

    @testset "noninf_mean" begin
        stats_input = [1.0, 2.0, Inf, 3.0]
        JR2K.noninf_stats(stats_input) # compile before measuring
        @test @allocated(JR2K.noninf_stats(stats_input)) == 0
        @test JR2K.noninf_mean([1.0, 2.0, Inf, 3.0]) ≈ 2.0
        @test isnan(JR2K.noninf_mean([1.0, NaN, 3.0]))  # NaN must propagate, not be filtered
        @test_throws ArgumentError JR2K.noninf_mean([Inf, Inf])
    end

    @testset "fssa_penalty_floor" begin
        di_center = (0.1, 0.2)
        dt = 0.5
        ρg_v = reshape(collect(1.0:12.0), 3, 4)  # constant column-to-column step of 3

        @test JR2K.fssa_penalty_floor(nothing, di_center, dt, 2, 2) == false

        for j in 1:4
            @test JR2K.fssa_penalty_floor(ρg_v, di_center, dt, 2, j) ≈ 3.0 * dt * di_center[2] / 2
        end
    end

    @testset "value_span / nonzero_span / masked_value_span" begin
        @test JR2K.value_span([1.0, 5.0, 3.0]) == 4.0
        @test JR2K.value_span([2.0, 2.0]) == 0.0
        @test JR2K.value_scale([-5.0, -4.0]) == 5.0
        @test JR2K.value_scale([1.0, 5.0, 3.0]) == 5.0
        @test JR2K.nonzero_span(0.0) == 1.0
        @test JR2K.nonzero_span(4.0) == 4.0

        mask = [true, false, true, true]
        A = [1.0, 100.0, 5.0, 2.0]
        JR2K.masked_extrema(mask, A) # compile before measuring
        @test @allocated(JR2K.masked_extrema(mask, A)) == 0
        @test JR2K.masked_value_span(mask, A) == 4.0  # span over {1.0, 5.0, 2.0}
        @test JR2K.masked_value_scale(mask, A) == 5.0
        @test JR2K.masked_value_span([false, false], [1.0, 2.0]) == 0.0  # empty mask ⇒ 0
    end

    @testset "Rayleigh quotient guard" begin
        @test JR2K.rayleigh_quotient(-6.0, 3.0) == 2.0
        @test JR2K.rayleigh_quotient(1.0, 0.0) == 0.0
    end

    @testset "reduced variational operator" begin
        if backend_JR === CPUBackend
            λmin = Float64[]
            for phi_case in (:full, :cut)
                A, dofs, Dfields, bounds_fields = reduced_velocity_matrix(phi_case)
                @test A ≈ A' atol = 1.0e-12
                @test rank(A) == size(A, 1)
                push!(λmin, minimum(eigvals(Symmetric(A))))

                D = [Dfields[d][I] for (d, I) in dofs]
                bounds = [bounds_fields[d][I] for (d, I) in dofs]
                exact_rows = vec(sum(abs, Diagonal(inv.(D)) * A; dims = 2))
                @test all(exact_rows .<= bounds .+ 100eps(maximum(bounds)))
            end
            # The cut domain is nonsingular but genuinely less well conditioned.
            @test 0 < λmin[2] < λmin[1]
        end
    end

    @testset "zero_nonfinite_ρg!" begin
        ρgy = PTArray(backend_JR)([1.0 Inf NaN; -2.0 0.0 3.0])
        @parallel (@idx size(ρgy)) JR2K.zero_nonfinite_ρg!(ρgy)
        @test Array(ρgy) == [1.0 0.0 0.0; -2.0 0.0 3.0]
    end

    @testset "compute_bulk_viscosity_and_penalty! FSSA floor" begin
        nx, ny = 4, 4
        ni = nx, ny
        li = 1.0, 1.0
        grid = Geometry(ni, li; origin = (0.0, 0.0))
        di = grid.di
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

        # baseline: no buoyancy field ⇒ physical γ_eff only (existing behavior)
        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, di, dt)
        γ_baseline = copy(Array(dyrel.γ_eff))

        # flat buoyancy ⇒ zero floor ⇒ γ_eff unchanged from the physical value
        flat_ρg = PTArray(backend_JR)(fill(5.0, ni...))
        JR2K.compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, dyrel.γfact, dt, di.center, flat_ρg)
        @test Array(dyrel.γ_eff) ≈ γ_baseline

        # In the variational formulation γ is an algorithmic penalty on the retained RP=0
        # constraint. The momentum gradient supplies the single cell-volume weight; γ itself
        # must therefore remain unweighted or the augmented block would scale as ϕ².
        ϕ = RockRatio(backend_JR, ni)
        for A in (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)
            A .= 0.5
        end
        JR2K.compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, dyrel.γfact, dt, di.center, flat_ρg)
        @test Array(dyrel.γ_eff) ≈ γ_baseline

        # steep buoyancy gradient ⇒ the FSSA floor exceeds the physical value everywhere
        steep_ρg = PTArray(backend_JR)([100.0 * j for i in 1:nx, j in 1:ny])
        JR2K.compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, dyrel.γfact, dt, di.center, steep_ρg)
        @test all(Array(dyrel.γ_eff) .> γ_baseline)
    end
end
