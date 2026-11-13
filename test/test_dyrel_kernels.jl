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
        JR2K.compute_stress_viscosity_DYREL!(
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
    end
end

@testset "Variational DYREL diagonal matches refined momentum operator" begin
    ni = (6, 5)
    xv = [0.0, 0.05, 0.12, 0.3, 0.5, 0.7, 1.0]
    yv = [0.0, 0.1, 0.15, 0.4, 0.8, 1.0]
    grid = Geometry(PTArray(backend), xv, yv; distributed = false)
    el = ConstantElasticity(; G = 3.0, Kb = 5.0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((LinearViscous(; η = 2.0), el)),
            Elasticity = el,
        ),
    )
    phases = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx ni) _init_single_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_single_phase!(phases.vertex)
    stokes = StokesArrays(backend, ni)
    stokes.viscosity.η .= 2.0
    ϕ = RockRatio(backend, ni)
    foreach(A -> fill!(A, 0.6), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
    dt = 0.7
    dyrel = DYREL(backend, stokes, rheology, phases, ϕ, grid.di, dt)
    ρg = @zeros(ni...), @zeros(ni...)
    θc = similar(stokes.P)
    i, j = 3, 3
    # A unit velocity at the row's own unknown extracts its diagonal from the actual
    # Schur-complement residual, independently of the Gershgorin coefficient formulas.
    function row_residual(d, component, ii, jj)
        foreach(A -> fill!(A, 0.0), @velocity(stokes))
        V = @velocity(stokes)[component]
        basis = zeros(size(V))
        basis[ii, jj] = 1.0
        V .= PTArray(backend)(basis)
        JR2K.compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phases, ϕ, grid._di, ni, dt)
        JR2K.compute_stress_DYREL!(stokes, rheology, phases, ϕ, 1.0, dt)
        @. θc = dyrel.γ_eff * stokes.R.RP
        @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, stokes.V.Vx, stokes.V.Vy, stokes.P, θc,
            stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg..., ϕ,
            grid._di.center, grid._di.vertex, 0.0,
        )
        return Array((stokes.R.Rx, stokes.R.Ry)[d])[i, j]
    end
    for d in 1:2
        diagonal = Array((dyrel.Dx, dyrel.Dy)[d])[i, j]
        @test diagonal ≈ -row_residual(d, d, i + 1, j + 1)
        row_sum = sum(abs(row_residual(d, c, ii, jj)) for c in 1:2, ii in i:(i + 2), jj in j:(j + 2))
        bound = Array((dyrel.λmaxVx, dyrel.λmaxVy)[d])[i, j] * diagonal
        @test bound ≈ row_sum
    end
end

@testset "Variational FSSA never lowers the Vy diagonal" begin
    ni = nx, ny = (6, 5)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    rheology = (SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((LinearViscous(; η = 2.0),))),)
    phases = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx ni) _init_single_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_single_phase!(phases.vertex)
    stokes = StokesArrays(backend, ni)
    stokes.viscosity.η .= 2.0
    stokes.V.Vy .= PTArray(backend)(rand(size(stokes.V.Vy)...))
    ϕ = RockRatio(backend, ni)
    foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
    dt = 0.7
    dyrel = DYREL(backend, stokes, rheology, phases, ϕ, grid.di, dt)
    JR2K.Gershgorin_variational!(Val(2), dyrel, stokes, phases, ϕ, rheology, grid.di, dt, nothing)
    Dy_viscous = Array(dyrel.Dy)
    ρgx = @zeros(ni...)
    function Ry_fssa(ρgy)
        Ry(dt) = (
            @parallel (@idx ni) JR2K.compute_PH_residual_V!(
                stokes.R.Rx, stokes.R.Ry, stokes.V.Vx, stokes.V.Vy, stokes.P, stokes.ΔPψ,
                stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρgx, ρgy, ϕ,
                grid._di.center, grid._di.vertex, dt,
            ); Array(stokes.R.Ry)
        )
        return Ry(dt) .- Ry(0.0)
    end
    Vy = Array(stokes.V.Vy)[2:(end - 1), 2:(end - 1)]
    # density decreasing upward, as across the rock/air interface: the term stabilizes
    ∂ρg∂y = -ny
    ρgy = PTArray(backend)([3.0 + ∂ρg∂y * y for _ in 1:nx, y in grid.xci[2]])
    JR2K.Gershgorin_variational!(Val(2), dyrel, stokes, phases, ϕ, rheology, grid.di, dt, ρgy)
    @test Array(dyrel.Dy) ≈ Dy_viscous .- ∂ρg∂y * dt
    @test Ry_fssa(ρgy) ≈ Vy .* (∂ρg∂y * dt)
    # density increasing upward, as in a Rayleigh–Taylor-unstable layer: the term is left out
    ρgy = PTArray(backend)([3.0 + ny * y for _ in 1:nx, y in grid.xci[2]])
    JR2K.Gershgorin_variational!(Val(2), dyrel, stokes, phases, ϕ, rheology, grid.di, dt, ρgy)
    @test Array(dyrel.Dy) == Dy_viscous
    @test all(iszero, Ry_fssa(ρgy))
end

@testset "Variational DYREL damping recovers a single mode" begin
    n = 8
    ni = n, n
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    bcs = VelocityBoundaryConditions(; free_slip = (left = false, right = false, top = false, bot = false), periodic = (left = true, right = true, top = true, bot = true))
    stokes = StokesArrays(backend, ni, bcs)
    rheology = (SetMaterialParams(; Phase = 1, Density = ConstantDensity(; ρ = 1.0), Gravity = ConstantGravity(; g = 0.0), CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),))),)
    phases = PhaseRatios(backend_JP, 1, ni)
    @parallel (@idx ni) _init_single_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_single_phase!(phases.vertex)
    ϕ = RockRatio(backend, ni)
    foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
    stokes.V.Vx .= PTArray(backend)([sinpi(2 * x) for x in grid.xvi[1], _ in 1:(n + 2)])
    stokes.viscosity.η .= 1.0
    dyrel = DYREL(backend, stokes, rheology, phases, ϕ, grid.di, Inf; ϵ = 1.0e-12)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P)
    igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    # One PH pass leaves the first damping estimate available for inspection.
    solve_VariationalDYREL!(
        stokes, (@zeros(ni...), @zeros(ni...)), dyrel, bcs, phases, ϕ, rheology, args, grid, Inf, igg;
        linear_viscosity = true, verbose_PH = false, verbose_DR = false,
        iterMax_PH = 1, iterMax_DR = 50, total_iterMax = 50, nout = 50
    )
    # This sinusoidal velocity is an eigenvector of the uniform Schur operator.
    diagonal = Array(dyrel.Dx)[1, 1]
    γ = Array(dyrel.γ_eff)[1, 1]
    λ = (γ + 4 / 3) * 4 * sinpi(1 / n)^2 * n^2 / diagonal
    @test all(isapprox.(Array(dyrel.cVx), 2 * sqrt(λ) * dyrel.c_fact; rtol = 1.0e-10))
    finalize_global_grid()
end
