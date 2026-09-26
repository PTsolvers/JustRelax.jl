push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using LinearAlgebra, Statistics, StaticArrays
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

# kernels launched with `@parallel` must come from the module compiled for the active backend
const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

const JR2 = JustRelax.JustRelax2D

dev(A) = PTArray(backend)(A)

# ---------------------------------------------------------------------------------------
# Reference constitutive relations
# ---------------------------------------------------------------------------------------

# plane-strain second invariant, with τzz = -(τxx + τyy)
τII_ps(xx, yy, xy) = sqrt(0.5 * (xx^2 + yy^2 + (xx + yy)^2) + xy^2)
# visco-elastic Maxwell viscosity and stress after one backward-Euler time step
ηve(η, G, dt) = inv(inv(η) + inv(G * dt))
maxwell(ε, τo, η, G, dt) = 2 * ηve(η, G, dt) * (ε + τo / (2 * G * dt))
# one implicit pseudo-time step θ (τⁿ⁺¹ - τⁿ) = 2ηε - η (τⁿ⁺¹ - τo) / (G dt) - τⁿ⁺¹
pt_step(τ, τo, ε, η, G, dt, θ) = (θ * τ + 2η * ε + η * τo / (G * dt)) / (θ + 1 + η / (G * dt))

# cell centers adjacent to vertex (i, j) that exist on a grid of `ni` cells
vertex_cells(ni, i, j) =
    [(ic, jc) for ic in (i - 1, i), jc in (j - 1, j) if 1 ≤ ic ≤ ni[1] && 1 ≤ jc ≤ ni[2]]
vmean(A, i, j) = mean(A[c...] for c in vertex_cells(size(A), i, j))
vharm(A, i, j) = (cs = vertex_cells(size(A), i, j); length(cs) / sum(inv(A[c...]) for c in cs))
# mean of the four vertices of cell (i, j)
cmean(A, i, j) = mean(@view A[i:(i + 1), j:(j + 1)])

# vertex coordinates of a smoothly stretched 1D grid of `n` cells spanning [x0, x0 + L]
function stretched(n, L, amp, phase; x0 = 0.0)
    w = [1 + amp * sin(k + phase) for k in 1:n]
    return x0 .+ vcat(0.0, cumsum(w)) .* (L / sum(w))
end

@parallel_indices (I...) function set_two_phases!(phases, w)
    @index phases[1, I...] = w[I...]
    @index phases[2, I...] = 1.0 - w[I...]
    return nothing
end

function two_phase_ratios(w_center, w_vertex)
    ni = size(w_center)
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    @parallel (@idx ni) set_two_phases!(phase_ratios.center, dev(w_center))
    @parallel (@idx ni .+ 1) set_two_phases!(phase_ratios.vertex, dev(w_vertex))
    return phase_ratios
end

@testset "Stokes kernels 2D" begin

    @testset "stress increment helpers" begin
        η, G, dt = 3.0, 2.0, 0.4
        _Gdt = inv(G * dt)
        τ, τo, ε = (0.7, -0.2, 0.35), (0.1, 0.4, -0.3), (1.2, -0.5, 0.8)

        # with θ_dτ = 0 the pseudo-time step lands on the Maxwell stress from any τ
        dτ_r = JR2.compute_dτ_r(0.0, η, _Gdt)
        for k in 1:3
            dτ = JR2.compute_stress_increment(τ[k], τo[k], η, ε[k], _Gdt, dτ_r)
            @test τ[k] + dτ ≈ maxwell(ε[k], τo[k], η, G, dt)
        end
        dτ = JR2.compute_stress_increment(τ, τo, η, ε, _Gdt, dτ_r)
        @test all(τ .+ dτ .≈ maxwell.(ε, τo, η, G, dt))

        # θ_dτ > 0 relaxes towards the Maxwell stress by one implicit pseudo-time step
        θ = 1.7
        dτ_r = JR2.compute_dτ_r(θ, η, _Gdt)
        dτ = JR2.compute_stress_increment(τ, τo, η, ε, _Gdt, dτ_r)
        @test all(τ .+ dτ .≈ pt_step.(τ, τo, ε, η, G, dt, θ))

        # strain-increment form: Δε = ε dt with the kernel's rescaled pseudo-time step
        dτ_r_inc = inv(θ * dt + η / G + dt)
        Δε = ε .* dt
        for k in 1:3
            dτk = JR2.compute_stress_increment(τ[k], τo[k], η, Δε[k], inv(G), dτ_r_inc, dt)
            @test τ[k] + dτk ≈ pt_step(τ[k], τo[k], ε[k], η, G, dt, θ)
        end
        dτ = JR2.compute_stress_increment(τ, τo, η, Δε, inv(G), dτ_r_inc, dt)
        @test all(τ .+ dτ .≈ pt_step.(τ, τo, ε, η, G, dt, θ))

        # trial stress and its plane-strain invariant
        dτ, τII_trial = JR2.compute_stress_increment_and_trial(τ, τo, η, ε, _Gdt, dτ_r)
        τ_trial = pt_step.(τ, τo, ε, η, G, dt, θ)
        @test all(τ .+ dτ .≈ τ_trial)
        @test τII_trial ≈ τII_ps(τ_trial...)

        # Drucker-Prager return: with λ at its converged value F / (η dτ_r) the corrected
        # stress keeps the trial direction and sits on the yield surface
        τy = 0.5 * τII_trial
        F = τII_trial - τy
        @test JR2.isyielding(true, τII_trial, τy)
        @test !JR2.isyielding(false, τII_trial, τy)
        @test !JR2.isyielding(true, τII_trial, 2τII_trial)
        λ_conv = F / (η * dτ_r)
        dτ_pl, λ, λdQdτ = JR2.compute_dτ_pl(τ, dτ, τy, τII_trial, η, λ_conv, 0.0, dτ_r, 0.0)
        τ_corr = τ .+ dτ_pl
        @test λ ≈ λ_conv
        @test τII_ps(τ_corr...) ≈ τy
        @test all(τ_corr ./ τy .≈ τ_trial ./ τII_trial)
        # plastic strain rate is λ ∂Q/∂τ with ∂Q/∂τij = τij / (2τII)
        @test all(λdQdτ .≈ λ .* τ_trial ./ (2 * τII_trial))

        # effective viscosity τII / (2εII), falling back to η for a rigid cell
        @test JR2.effective_viscosity(3.0, 0.5, 7.0) ≈ 3.0
        @test JR2.effective_viscosity(3.0, 0.0, 7.0) == 7.0
    end

    @testset "plastic parameters and yield function" begin
        pl1 = DruckerPrager_regularised(; C = 2.0, ϕ = 30.0, Ψ = 5.0, η_vp = 0.1)
        pl2 = DruckerPrager_regularised(; C = 5.0, ϕ = 10.0, Ψ = 0.0, η_vp = 0.3)
        visc = LinearViscous(; η = 1.0)
        el = ConstantElasticity(; G = 1.0, Kb = 2.0)
        rheology = (
            SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((visc, el, pl1))),
            SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((visc, el, pl2))),
            SetMaterialParams(; Phase = 3, CompositeRheology = CompositeRheology((visc, el))),
        )

        # phase-weighted plastic parameters are the ratio-weighted sums
        ratio = (0.25, 0.75, 0.0)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR2.plastic_params_phase(rheology, 0.0, ratio)
        @test is_pl
        @test C ≈ 0.25 * 2.0 + 0.75 * 5.0
        @test sinϕ ≈ 0.25 * sind(30) + 0.75 * sind(10)
        @test cosϕ ≈ 0.25 * cosd(30) + 0.75 * cosd(10)
        @test sinψ ≈ 0.25 * sind(5)
        @test η_reg ≈ 0.25 * 0.1 + 0.75 * 0.3
        # cohesion perturbation scales C only
        _, Cp, sinϕp = JR2.plastic_params_phase(rheology, 0.0, ratio, (; perturbation_C = 1.5))
        @test Cp ≈ 1.5 * C
        @test sinϕp ≈ sinϕ
        # a cell made only of the non-plastic phase never yields
        is_pl3, C3 = JR2.plastic_params_phase(rheology, 0.0, (0.0, 0.0, 1.0))
        @test !is_pl3
        @test C3 == 0.0

        # F = τII - C cosϕ - P sinϕ, and F = τII for a phase without plasticity
        P, τII = 1.3, 4.0
        F1 = τII - 2.0 * cosd(30) - P * sind(30)
        F2 = τII - 5.0 * cosd(10) - P * sind(10)
        @test JR2.compute_yieldfunction_phase(rheology, 1; P, τII, EII = 0.0) ≈ F1
        @test JR2.compute_yieldfunction_phase(rheology, 3; P, τII, EII = 0.0) ≈ τII
        @test JR2.compute_yieldfunction_phase(rheology, (0.5, 0.3, 0.2); P, τII, EII = 0.0) ≈
            0.5 * F1 + 0.3 * F2 + 0.2 * τII
        @test JR2.compute_yieldfunction_phase(rheology, SA[0.0, 1.0, 0.0]; P, τII, EII = 0.0) ≈ F2

        # ∂Q/∂τij = τij / (2τII) in tensor convention, ∂Q/∂P = -sinψ, ∂F/∂P = -sinϕ
        τij = (0.6, -0.2, 0.9)
        τII_ij = τII_ps(τij...)
        dQdτ, dQdP, dFdP = JR2.compute_plastic_gradients_phase(rheology, 1, τij; P, τII = τII_ij, EII = 0.0)
        @test all(dQdτ .≈ τij ./ (2τII_ij))
        @test dQdP ≈ -sind(5)
        @test dFdP ≈ -sind(30)
        dQdτ3, dQdP3, dFdP3 = JR2.compute_plastic_gradients_phase(rheology, 3, τij; P, τII = τII_ij, EII = 0.0)
        @test all(iszero, dQdτ3) && iszero(dQdP3) && iszero(dFdP3)
        dQdτw, dQdPw, dFdPw = JR2.compute_plastic_gradients_phase(
            rheology, (0.5, 0.5, 0.0), τij; P, τII = τII_ij, EII = 0.0
        )
        @test all(dQdτw .≈ τij ./ (2τII_ij))
        @test dQdPw ≈ -0.5 * sind(5)
        @test dFdPw ≈ -0.5 * (sind(30) + sind(10))
        dQdτs, = JR2.compute_plastic_gradients_phase(
            rheology, SA[0.0, 0.0, 1.0], τij; P, τII = τII_ij, EII = 0.0
        )
        @test all(iszero, dQdτs)
    end

    @testset "strain softening of plastic parameters" begin
        # LinearSoftening ramps a parameter linearly from its nominal value at EII = lo to
        # min_value at EII = hi, constant outside
        lo, hi = 0.1, 0.5
        ramp(v0, vmin, EII) = EII ≤ lo ? v0 : EII ≥ hi ? vmin : v0 + (vmin - v0) * (EII - lo) / (hi - lo)
        C0, Cmin, ϕ0, ϕmin, ψ, ηvp = 2.0, 0.5, 30.0, 15.0, 4.0, 0.3
        soft_C = LinearSoftening(Cmin, C0, lo, hi)
        soft_ϕ = LinearSoftening(ϕmin, ϕ0, lo, hi)
        visc = LinearViscous(; η = 1.0)
        mat(pl) = SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((visc, pl)))
        cases = (
            (DruckerPrager_regularised(; C = C0, ϕ = ϕ0, Ψ = ψ, η_vp = ηvp, softening_C = soft_C, softening_ϕ = soft_ϕ), true, true),
            (DruckerPrager_regularised(; C = C0, ϕ = ϕ0, Ψ = ψ, η_vp = ηvp, softening_C = soft_C), true, false),
            (DruckerPrager_regularised(; C = C0, ϕ = ϕ0, Ψ = ψ, η_vp = ηvp, softening_ϕ = soft_ϕ), false, true),
            (DruckerPragerCap(; C = C0, ϕ = ϕ0, Ψ = ψ, η_vp = ηvp, pT = -1.0), false, false),
            (DruckerPragerCap(; C = C0, ϕ = ϕ0, Ψ = ψ, η_vp = ηvp, pT = -1.0, softening_C = soft_C, softening_ϕ = soft_ϕ), true, true),
        )
        for (pl, softC, softϕ) in cases, EII in (0.0, 0.3, 0.8)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR2.plastic_params(mat(pl), EII)
            ϕ = softϕ ? ramp(ϕ0, ϕmin, EII) : ϕ0
            @test is_pl
            @test C ≈ (softC ? ramp(C0, Cmin, EII) : C0)
            @test sinϕ ≈ sind(ϕ)
            @test cosϕ ≈ cosd(ϕ)
            @test sinψ ≈ sind(ψ)
            @test η_reg ≈ ηvp
        end
        # the softening helpers act on plain DruckerPrager as well
        for (pl, softC, softϕ) in (
                    (DruckerPrager(; C = C0, ϕ = ϕ0), false, false),
                    (DruckerPrager(; C = C0, ϕ = ϕ0, softening_C = soft_C, softening_ϕ = soft_ϕ), true, true),
                ), EII in (0.0, 0.3)
            @test JR2.soften_cohesion(pl, EII) ≈ (softC ? ramp(C0, Cmin, EII) : C0)
            @test all(JR2.soften_friction_angle(pl, EII) .≈ sincosd(softϕ ? ramp(ϕ0, ϕmin, EII) : ϕ0))
        end
        # the one-argument form evaluates the unsoftened parameters
        @test JR2.plastic_params(mat(cases[1][1]))[2] ≈ C0
        @test JR2.plastic_params(mat(visc)) == (false, 0.0, 0.0, 0.0, 0.0, 0.0)
    end

    @testset "tensor caching at cell centers" begin
        n1, n2 = 3, 4
        A(k) = [k + 0.1 * i + 0.01 * j for i in 1:n1, j in 1:n2]
        Av(k) = [k + 0.1 * i + 0.01 * j^2 for i in 1:(n1 + 1), j in 1:(n2 + 1)]
        τ, τo, ε = (A(1), A(2), A(3)), (A(4), A(5), A(6)), (A(7), A(8), Av(9))
        i, j = 2, 3
        τij, τij_o, εij = JR2.cache_tensors(τ, τo, ε, i, j)
        @test τij == getindex.(τ, i, j)
        @test τij_o == getindex.(τo, i, j)
        # shear strain rate lives on vertices and is averaged onto the center
        @test all(εij .≈ (ε[1][i, j], ε[2][i, j], cmean(ε[3], i, j)))

        τw = (zeros(n1, n2), zeros(n1, n2), zeros(n1, n2))
        JR2.correct_stress!(τw..., (1.5, -2.0, 0.25), i, j)
        @test getindex.(τw, i, j) == (1.5, -2.0, 0.25)
        @test sum(sum, τw) ≈ 1.5 - 2.0 + 0.25
    end

    # anisotropic, non-uniform grid shared by the kernel tests
    nx, ny = 7, 5
    ni = nx, ny
    xv = stretched(nx, 2.0, 0.6, 0.3)
    yv = stretched(ny, 1.0, 0.5, 1.1; x0 = -1.0)
    grid = Geometry(PTArray(backend), xv, yv)
    xc = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2
    yc = (yv[1:(end - 1)] .+ yv[2:end]) ./ 2
    xvx, yvx = Array.(grid.xi_vel[1])
    xvy, yvy = Array.(grid.xi_vel[2])

    @testset "divergence and strain rate" begin
        stokes = StokesArrays(backend, ni)
        # V = A x + v0 has constant ∇V = A11 + A22 and ε = sym(A) - ∇V/3 I on any grid
        for (a, b, c, d) in ((0.8, -0.3, 1.1, -0.45), (0.0, 0.7, -0.7, 0.0))
            v0 = (0.4, -1.3)
            copyto!(stokes.V.Vx, [v0[1] + a * x + b * y for x in xvx, y in yvx])
            copyto!(stokes.V.Vy, [v0[2] + c * x + d * y for x in xvy, y in yvy])

            @parallel (@idx ni) JR2K.compute_∇V!(stokes.∇V, @velocity(stokes), grid._di.vertex)
            @parallel (@idx ni .+ 1) JR2K.compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)...,
                grid._di.vertex, grid._di.velocity[1], grid._di.velocity[2],
            )
            divV = a + d
            @test all(isapprox.(Array(stokes.∇V), divV; atol = 1.0e-12))
            @test all(isapprox.(Array(stokes.ε.xx), a - divV / 3; atol = 1.0e-12))
            @test all(isapprox.(Array(stokes.ε.yy), d - divV / 3; atol = 1.0e-12))
            @test all(isapprox.(Array(stokes.ε.xy), (b + c) / 2; atol = 1.0e-12))
        end
    end

    # non-uniform fields for the constitutive kernels
    η_c = [1.0 + 0.5 * sin(2i + j) for i in 1:nx, j in 1:ny]
    G_c = [0.8 + 0.3 * cos(i - 2j) for i in 1:nx, j in 1:ny]
    εxx_c = [0.3 * sin(i) + 0.1 * j for i in 1:nx, j in 1:ny]
    εyy_c = [-0.2 * cos(j) + 0.05 * i for i in 1:nx, j in 1:ny]
    εxy_v = [0.25 * sin(i * j) + 0.1 for i in 1:(nx + 1), j in 1:(ny + 1)]
    τxx_c = [0.1 * i - 0.2 * j for i in 1:nx, j in 1:ny]
    τyy_c = [0.05 * i * j - 0.3 for i in 1:nx, j in 1:ny]
    τxy_v = [0.2 * cos(i + j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    τxx_o = [0.15 * cos(i) for i in 1:nx, j in 1:ny]
    τyy_o = [-0.1 * sin(j) for i in 1:nx, j in 1:ny]
    τxy_ov = [0.05 * (i - j) for i in 1:(nx + 1), j in 1:(ny + 1)]
    dt = 0.7

    @testset "visco-elastic compute_τ! (G field)" begin
        θ = 0.9
        τxx, τyy, τxy = dev(τxx_c), dev(τyy_c), dev(τxy_v)
        @parallel (@idx ni .+ 1) JR2K.compute_τ!(
            τxx, τyy, τxy, dev(τxx_o), dev(τyy_o), dev(τxy_ov),
            dev(εxx_c), dev(εyy_c), dev(εxy_v), dev(η_c), dev(G_c), θ, dt,
        )
        @test Array(τxx) ≈ pt_step.(τxx_c, τxx_o, εxx_c, η_c, G_c, dt, θ)
        @test Array(τyy) ≈ pt_step.(τyy_c, τyy_o, εyy_c, η_c, G_c, dt, θ)
        # shear stress on vertices uses the arithmetic mean of the adjacent cells' η and G
        τxy_ref = [
            pt_step(τxy_v[i, j], τxy_ov[i, j], εxy_v[i, j], vmean(η_c, i, j), vmean(G_c, i, j), dt, θ)
                for i in 1:(nx + 1), j in 1:(ny + 1)
        ]
        @test Array(τxy) ≈ τxy_ref
    end

    @testset "compute_τ! with phase ratios and compute_τ_vertex!" begin
        θ = 0.4
        G1, G2 = 1.5, 0.6
        rheology = (
            SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), ConstantElasticity(; G = G1, Kb = 1.0)))),
            SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), ConstantElasticity(; G = G2, Kb = 1.0)))),
        )
        w_c = [0.5 + 0.5 * sin(i + 3j) for i in 1:nx, j in 1:ny]
        w_v = [0.5 + 0.5 * cos(i - j) for i in 1:(nx + 1), j in 1:(ny + 1)]
        phase_ratios = two_phase_ratios(w_c, w_v)
        Gmix = @. w_c * G1 + (1 - w_c) * G2

        τxy_c = [0.1 * i - 0.05 * j for i in 1:nx, j in 1:ny]
        τxy_oc = [0.02 * i * j for i in 1:nx, j in 1:ny]
        τxx, τyy, τxy = dev(τxx_c), dev(τyy_c), dev(τxy_c)
        @parallel (@idx ni) JR2K.compute_τ!(
            τxx, τyy, τxy, dev(τxx_o), dev(τyy_o), dev(τxy_oc),
            dev(εxx_c), dev(εyy_c), dev(εxy_v), dev(η_c), θ, dt,
            phase_ratios.center, rheology,
        )
        εxy_c = [cmean(εxy_v, i, j) for i in 1:nx, j in 1:ny]
        @test Array(τxx) ≈ pt_step.(τxx_c, τxx_o, εxx_c, η_c, Gmix, dt, θ)
        @test Array(τyy) ≈ pt_step.(τyy_c, τyy_o, εyy_c, η_c, Gmix, dt, θ)
        @test Array(τxy) ≈ pt_step.(τxy_c, τxy_oc, εxy_c, η_c, Gmix, dt, θ)

        # purely viscous vertex update with the harmonic mean of the four cells' viscosity
        τxy = dev(τxy_v)
        @parallel (@idx ni .+ 1) JR2K.compute_τ_vertex!(τxy, dev(εxy_v), dev(η_c), θ)
        τxy_h = Array(τxy)
        for j in 1:(ny + 1), i in 1:(nx + 1)
            if 2 ≤ i ≤ nx && 2 ≤ j ≤ ny
                ηv = 4 / sum(inv, η_c[(i - 1):i, (j - 1):j])
                @test τxy_h[i, j] ≈ (θ * τxy_v[i, j] + 2ηv * εxy_v[i, j]) / (θ + 1)
            else
                @test τxy_h[i, j] == τxy_v[i, j]
            end
        end
    end

    # Visco-elasto-plastic Drucker-Prager rheology. With θ_dτ = 0 the trial stress is the
    # Maxwell stress τM = 2ηve ε_ve, ε_ve = ε + τo / (2G dt), so repeated kernel calls only
    # relax the plastic multiplier λ to its converged value. The converged state satisfies
    #   τII = C cosϕ + Pc sinϕ + η_vp λ,   Pc = P + K dt λ sinψ     (yield at augmented P)
    #   τ ∥ τM                                                     (associated deviatoric flow)
    #   ε_pl = ε_ve - τ / (2ηve),  εII_pl = λ / 2                  (strain-rate partition)
    C1, ϕ1, ψ1, ηvp1, G1, K1 = 0.12, 30.0, 10.0, 0.05, 1.0, 3.0
    C2, ϕ2, ψ2, ηvp2, G2, K2 = 0.3, 20.0, 0.0, 0.2, 0.7, 5.0
    pl1 = DruckerPrager_regularised(; C = C1, ϕ = ϕ1, Ψ = ψ1, η_vp = ηvp1)
    pl2 = DruckerPrager_regularised(; C = C2, ϕ = ϕ2, Ψ = ψ2, η_vp = ηvp2)
    visc = LinearViscous(; η = 1.0)
    rheo_pl = (
        SetMaterialParams(; Phase = 1, CompositeRheology = CompositeRheology((visc, ConstantElasticity(; G = G1, Kb = K1), pl1))),
        SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((visc, ConstantElasticity(; G = G2, Kb = K2), pl2))),
    )
    rheo_mix = (
        rheo_pl[1],
        SetMaterialParams(; Phase = 2, CompositeRheology = CompositeRheology((visc, ConstantElasticity(; G = G2, Kb = K2)))),
    )
    # P keeps C cosϕ + P sinϕ > 0: no cell is in the tensile regime
    P_c = [0.15 * sin(i + j) + 0.02 * i + 0.1 for i in 1:nx, j in 1:ny]
    εxy_c = [cmean(εxy_v, i, j) for i in 1:nx, j in 1:ny]
    τxy_oc = [0.03 * (i - 2j) for i in 1:nx, j in 1:ny]
    # pure-phase cells in a checkerboard-like pattern; vertices carry their own pattern
    w_c = [Float64(isodd(i + 2j)) for i in 1:nx, j in 1:ny]
    w_v = [Float64(isodd(i * j)) for i in 1:(nx + 1), j in 1:(ny + 1)]

    function check_dp_state(τ, λ, Pc, εpl, εij, τo, η, G, K, dt, C, ϕ, ψ, ηvp, P, plastic)
        ε_ve = εij .+ τo ./ (2G * dt)
        ηe = ηve(η, G, dt)
        τM = 2ηe .* ε_ve
        yields = plastic && τII_ps(τM...) > C * cosd(ϕ) + P * sind(ϕ)
        if yields
            τII = τII_ps(τ...)
            @test λ > 0
            @test Pc ≈ P + K * dt * λ * sind(ψ)
            @test τII ≈ C * cosd(ϕ) + Pc * sind(ϕ) + ηvp * λ
            @test all(τ ./ τII .≈ τM ./ τII_ps(τM...))
            @test all(isapprox.(εpl, ε_ve .- τ ./ (2ηe); atol = 1.0e-12))
            @test τII_ps(εpl...) ≈ λ / 2
        else
            @test λ == 0
            @test Pc ≈ P
            @test all(τ .≈ τM)
            @test all(abs.(εpl) .< 1.0e-14)
        end
        return yields
    end

    @testset "compute_τ_nonlinear! (single and multi phase)" begin
        for multiphase in (false, true)
            rheology = multiphase ? rheo_pl : (rheo_pl[1],)
            stokes = StokesArrays(backend, ni)
            copyto!(stokes.ε.xx, εxx_c); copyto!(stokes.ε.yy, εyy_c); copyto!(stokes.ε.xy, εxy_v)
            copyto!(stokes.τ_o.xx, τxx_o); copyto!(stokes.τ_o.yy, τyy_o); copyto!(stokes.τ_o.xy_c, τxy_oc)
            copyto!(stokes.P, P_c)
            η, θ, λ = dev(η_c), @zeros(ni...), @zeros(ni...)
            phase_ratios = two_phase_ratios(w_c, w_v)
            args = (; T = @zeros(ni...))
            for _ in 1:80
                if multiphase
                    @parallel (@idx ni) JR2K.compute_τ_nonlinear!(
                        @tensor_center(stokes.τ), stokes.τ.II, @tensor_center(stokes.τ_o),
                        @strain(stokes), @plastic_strain(stokes), stokes.EII_pl,
                        stokes.ε_vol_pl, stokes.EVol_pl, stokes.P, θ, η, stokes.viscosity.η_vep,
                        λ, phase_ratios.center, rheology, dt, 0.0, args,
                    )
                else
                    @parallel (@idx ni) JR2K.compute_τ_nonlinear!(
                        @tensor_center(stokes.τ), stokes.τ.II, @tensor_center(stokes.τ_o),
                        @strain(stokes), @plastic_strain(stokes), stokes.EII_pl,
                        stokes.P, θ, η, stokes.viscosity.η_vep, λ, rheology, dt, 0.0, args,
                    )
                end
            end
            τ_h = Array.(@tensor_center(stokes.τ))
            εpl_h = Array.((stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy))
            λ_h, θ_h, τII_h, ηvep_h = Array(λ), Array(θ), Array(stokes.τ.II), Array(stokes.viscosity.η_vep)
            nyield = 0
            for j in 1:ny, i in 1:nx
                phase1 = !multiphase || w_c[i, j] == 1
                G, K, C, ϕ, ψ, ηvp = phase1 ? (G1, K1, C1, ϕ1, ψ1, ηvp1) : (G2, K2, C2, ϕ2, ψ2, ηvp2)
                εij = (εxx_c[i, j], εyy_c[i, j], εxy_c[i, j])
                τij = getindex.(τ_h, i, j)
                εpl = getindex.(εpl_h, i, j)
                nyield += check_dp_state(
                    τij, λ_h[i, j], θ_h[i, j], εpl, εij, (τxx_o[i, j], τyy_o[i, j], τxy_oc[i, j]),
                    η_c[i, j], G, K, dt, C, ϕ, ψ, ηvp, P_c[i, j], true,
                )
                @test τII_h[i, j] ≈ τII_ps(τij...)
                @test ηvep_h[i, j] ≈ τII_h[i, j] / (2 * τII_ps(εij...))
            end
            # the field is chosen so that both yielding and non-yielding cells occur
            @test 0 < nyield < nx * ny
        end
    end

    @testset "update_stresses_center_vertex_ps!" begin
        relλ = 0.5
        function run_ps!(stokes, λ, λv, Pc, phase_ratios, rheology, n; increment = false)
            for _ in 1:n
                if increment
                    @parallel (@idx ni .+ 1) JR2K.update_stresses_center_vertex_ps!(
                        @strain(stokes), @strain_increment(stokes), @plastic_strain(stokes),
                        stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                        @tensor_center(stokes.τ), (stokes.τ.xy,), @tensor_center(stokes.τ_o), (stokes.τ_o.xy,),
                        stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                        relλ, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                    )
                else
                    @parallel (@idx ni .+ 1) JR2K.update_stresses_center_vertex_ps!(
                        @strain(stokes), @plastic_strain(stokes),
                        stokes.EII_pl, stokes.ε_vol_pl, stokes.EVol_pl,
                        @tensor_center(stokes.τ), (stokes.τ.xy,), @tensor_center(stokes.τ_o), (stokes.τ_o.xy,),
                        stokes.P, Pc, stokes.viscosity.η, λ, λv, stokes.τ.II, stokes.viscosity.η_vep,
                        relλ, dt, 0.0, rheology, phase_ratios.center, phase_ratios.vertex,
                    )
                end
            end
            return nothing
        end
        function setup_ps()
            stokes = StokesArrays(backend, ni)
            copyto!(stokes.ε.xx, εxx_c); copyto!(stokes.ε.yy, εyy_c); copyto!(stokes.ε.xy, εxy_v)
            copyto!(stokes.Δε.xx, εxx_c .* dt); copyto!(stokes.Δε.yy, εyy_c .* dt); copyto!(stokes.Δε.xy, εxy_v .* dt)
            copyto!(stokes.τ_o.xx, τxx_o); copyto!(stokes.τ_o.yy, τyy_o)
            copyto!(stokes.τ_o.xy_c, τxy_oc); copyto!(stokes.τ_o.xy, τxy_ov)
            copyto!(stokes.P, P_c)
            copyto!(stokes.viscosity.η, η_c)
            return stokes, @zeros(ni...), @zeros(ni .+ 1...), @zeros(ni...)
        end

        for rheology in (rheo_pl, rheo_mix)
            plastic2 = rheology === rheo_pl
            phase_ratios = two_phase_ratios(w_c, w_v)
            stokes, λ, λv, Pc = setup_ps()
            run_ps!(stokes, λ, λv, Pc, phase_ratios, rheology, 80)

            τ_h = Array.(@tensor_center(stokes.τ))
            εpl_h = Array.((stokes.ε_pl.xx, stokes.ε_pl.yy))
            λ_h, Pc_h, εvol_h = Array(λ), Array(Pc), Array(stokes.ε_vol_pl)
            nyield = 0
            for j in 1:ny, i in 1:nx
                phase1 = w_c[i, j] == 1
                G, K, C, ϕ, ψ, ηvp = phase1 ? (G1, K1, C1, ϕ1, ψ1, ηvp1) : (G2, K2, C2, ϕ2, ψ2, ηvp2)
                εij = (εxx_c[i, j], εyy_c[i, j], εxy_c[i, j])
                τo = (τxx_o[i, j], τyy_o[i, j], τxy_oc[i, j])
                τij = getindex.(τ_h, i, j)
                ε_ve = εij .+ τo ./ (2G * dt)
                # normal plastic strain rates are stored at centers; xy lives on vertices
                εpl = (εpl_h[1][i, j], εpl_h[2][i, j], ε_ve[3] - τij[3] / (2ηve(η_c[i, j], G, dt)))
                y = check_dp_state(
                    τij, λ_h[i, j], Pc_h[i, j], εpl, εij, τo,
                    η_c[i, j], G, K, dt, C, ϕ, ψ, ηvp, P_c[i, j], phase1 || plastic2,
                )
                nyield += y
                # dilatant plastic flow: ε_vol_pl = -λ ∂Q/∂P = λ sinψ
                @test εvol_h[i, j] ≈ λ_h[i, j] * sind(ψ) atol = 1.0e-14
            end
            @test 0 < nyield < nx * ny

            # vertices: arithmetic means of P, normal strain rates and old normal stresses,
            # harmonic mean of η, material from the vertex phase
            τxy_h, λv_h, εplxy_h = Array(stokes.τ.xy), Array(λv), Array(stokes.ε_pl.xy)
            nyield_v = 0
            for j in 1:(ny + 1), i in 1:(nx + 1)
                phase1 = w_v[i, j] == 1
                G, K, C, ϕ, ψ, ηvp = phase1 ? (G1, K1, C1, ϕ1, ψ1, ηvp1) : (G2, K2, C2, ϕ2, ψ2, ηvp2)
                ηe = ηve(vharm(η_c, i, j), G, dt)
                ε_ve = (vmean(εxx_c, i, j), vmean(εyy_c, i, j), εxy_v[i, j]) .+
                    (vmean(τxx_o, i, j), vmean(τyy_o, i, j), τxy_ov[i, j]) ./ (2G * dt)
                τM = 2ηe .* ε_ve
                Pv = vmean(P_c, i, j)
                if (phase1 || plastic2) && τII_ps(τM...) > C * cosd(ϕ) + Pv * sind(ϕ)
                    nyield_v += 1
                    λv_ij = λv_h[i, j]
                    τII = τII_ps(τM...) * τxy_h[i, j] / τM[3]
                    @test τII ≈ C * cosd(ϕ) + (Pv + K * dt * λv_ij * sind(ψ)) * sind(ϕ) + ηvp * λv_ij
                    @test εplxy_h[i, j] ≈ ε_ve[3] - τxy_h[i, j] / (2ηe)
                    @test εplxy_h[i, j] ≈ λv_ij * τM[3] / (2 * τII_ps(τM...))
                else
                    @test λv_h[i, j] == 0
                    @test τxy_h[i, j] ≈ τM[3]
                    @test εplxy_h[i, j] == 0
                end
            end
            @test 0 < nyield_v < (nx + 1) * (ny + 1)

            # the strain-increment kernel with Δε = ε dt is the same update
            stokes2, λ2, λv2, Pc2 = setup_ps()
            run_ps!(stokes2, λ2, λv2, Pc2, phase_ratios, rheology, 80; increment = true)
            for (A, B) in zip(
                    (@tensor_center(stokes.τ)..., stokes.τ.xy, λ, λv, Pc, stokes.ε_pl.xx, stokes.ε_pl.xy, stokes.ε_vol_pl, stokes.viscosity.η_vep),
                    (@tensor_center(stokes2.τ)..., stokes2.τ.xy, λ2, λv2, Pc2, stokes2.ε_pl.xx, stokes2.ε_pl.xy, stokes2.ε_vol_pl, stokes2.viscosity.η_vep),
                )
                @test Array(A) ≈ Array(B) atol = 1.0e-13
            end
        end
    end

    @testset "pressure update" begin
        r, θ_dτ = 0.7, 0.2
        ∇V_c = [0.2 * sin(i) * cos(j) for i in 1:nx, j in 1:ny]
        Q_c = [0.01 * (i + j) for i in 1:nx, j in 1:ny]
        P0_c = [0.5 + 0.1 * cos(i * j) for i in 1:nx, j in 1:ny]
        Pini = [0.2 * sin(i - j) for i in 1:nx, j in 1:ny]
        K_c = [2.0 + sin(i + 2j) for i in 1:nx, j in 1:ny]
        dtP = 0.3
        ψ = @. r / θ_dτ * ηve(η_c, G_c, dtP)

        # one step solves (Pⁿ⁺¹ - Pⁿ) / ψ = -∇V - (Pⁿ⁺¹ - P0) / (K dt) + Q / dt
        P, RP = dev(Pini), @zeros(ni...)
        @parallel JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), dev(K_c), dev(G_c), dtP, r, θ_dτ)
        P1 = Array(P)
        @test Array(RP) ≈ @. -∇V_c - (Pini - P0_c) / (K_c * dtP) + Q_c / dtP
        @test (P1 .- Pini) ./ ψ ≈ @. -∇V_c - (P1 - P0_c) / (K_c * dtP) + Q_c / dtP
        # iterating converges to the compressible mass balance P = P0 + K (Q - dt ∇V)
        for _ in 1:300
            @parallel JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), dev(K_c), dev(G_c), dtP, r, θ_dτ)
        end
        @test Array(P) ≈ @. P0_c + K_c * (Q_c - dtP * ∇V_c)
        @test maximum(abs, Array(RP)) < 1.0e-12

        # K = G = ∞ reduces to the incompressible update P += r/θ_dτ η (Q/dt - ∇V)
        P, RP = dev(Pini), @zeros(ni...)
        Pinc, RPinc = dev(Pini), @zeros(ni...)
        Inf_c = fill(Inf, ni)
        @parallel JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), dev(Inf_c), dev(Inf_c), dtP, r, θ_dτ)
        @parallel JR2K.compute_P!(Pinc, RPinc, dev(∇V_c), dev(Q_c), dev(η_c), dtP, r, θ_dτ)
        @test Array(P) ≈ @. Pini + r / θ_dτ * η_c * (Q_c / dtP - ∇V_c)
        @test Array(Pinc) ≈ Array(P)
        @test Array(RPinc) ≈ Array(RP)

        # GeoParams bulk and shear moduli per phase
        α1, α2 = 3.0e-2, 1.0e-2
        rheo_P = (
            SetMaterialParams(; Phase = 1, Density = PT_Density(; ρ0 = 1.0, α = α1, β = 0.0, T0 = 0.0), CompositeRheology = CompositeRheology((visc, ConstantElasticity(; G = G1, Kb = K1)))),
            SetMaterialParams(; Phase = 2, Density = PT_Density(; ρ0 = 1.0, α = α2, β = 0.0, T0 = 0.0), CompositeRheology = CompositeRheology((visc, ConstantElasticity(; G = G2, Kb = K2)))),
        )
        phase_int = [w_c[i, j] == 1 ? 1 : 2 for i in 1:nx, j in 1:ny]
        Kp = [p == 1 ? K1 : K2 for p in phase_int]
        αp = [p == 1 ? α1 : α2 for p in phase_int]
        ΔT_c = [5.0 * sin(i + j) for i in 1:nx, j in 1:ny]
        P, RP = dev(Pini), @zeros(ni...)
        for _ in 1:300
            @parallel (@idx ni) JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), rheo_P, dev(phase_int), dtP, r, θ_dτ, (;))
        end
        @test Array(P) ≈ @. P0_c + Kp * (Q_c - dtP * ∇V_c)

        # phase-ratio form, isothermal and with thermal expansion P = P0 + K (Q - dt ∇V + α ΔT)
        phase_ratios = two_phase_ratios(w_c, w_v)
        P, RP = dev(Pini), @zeros(ni...)
        for _ in 1:300
            JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), rheo_P, phase_ratios, dtP, r, θ_dτ, (;))
        end
        @test Array(P) ≈ @. P0_c + Kp * (Q_c - dtP * ∇V_c)
        # a melt fraction without temperature change leaves the isothermal balance
        P, RP = dev(Pini), @zeros(ni...)
        for _ in 1:300
            JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), rheo_P, phase_ratios, dtP, r, θ_dτ, (; melt_fraction = @zeros(ni...)))
        end
        @test Array(P) ≈ @. P0_c + Kp * (Q_c - dtP * ∇V_c)
        for kw in ((; ΔT = dev(ΔT_c)), (; ΔT = dev(ΔT_c), melt_fraction = @zeros(ni...)))
            P, RP = dev(Pini), @zeros(ni...)
            for _ in 1:300
                JR2K.compute_P!(P, dev(P0_c), RP, dev(∇V_c), dev(Q_c), dev(η_c), rheo_P, phase_ratios, dtP, r, θ_dτ, kw)
            end
            @test Array(P) ≈ @. P0_c + Kp * (Q_c - dtP * ∇V_c + αp * ΔT_c)
        end
    end

    @testset "momentum residual and velocity update" begin
        # linear P and τ fields have exact staggered derivatives on the stretched grid;
        # ρgx varies along y and ρgy along x only, so their face averages are exact too
        p = (0.3, -0.8)
        sxx, syy, sxy = (1.1, 0.4), (-0.6, 0.9), (0.25, -1.3)
        Pf = [p[1] * x + p[2] * y for x in xc, y in yc]
        τxx = [sxx[1] * x + sxx[2] * y for x in xc, y in yc]
        τyy = [syy[1] * x + syy[2] * y for x in xc, y in yc]
        τxy = [sxy[1] * x + sxy[2] * y for x in xv, y in yv]
        ρgx = [0.5 + y^2 for x in xc, y in yc]
        ρgy = [1.0 - 0.3 * x for x in xc, y in yc]
        Rx_ref = [sxx[1] + sxy[2] - p[1] - (0.5 + y^2) for x in xv[2:(end - 1)], y in yc]
        Ry_ref = [syy[2] + sxy[1] - p[2] - (1.0 - 0.3 * x) for x in xc, y in yv[2:(end - 1)]]

        stokes = StokesArrays(backend, ni)
        fields = dev.((Pf, τxx, τyy, τxy, ρgx, ρgy))
        @parallel (@idx ni) JR2K.compute_Res!(
            stokes.R.Rx, stokes.R.Ry, fields..., grid._di.center, grid._di.vertex,
        )
        @test Array(stokes.R.Rx) ≈ Rx_ref
        @test Array(stokes.R.Ry) ≈ Ry_ref

        # free-surface form: ρgy independent of y gives no stabilization term
        Vx0 = [0.1 * sin(i + j) for i in 1:(nx + 1), j in 1:(ny + 2)]
        Vy0 = [0.2 * cos(i - j) for i in 1:(nx + 2), j in 1:(ny + 1)]
        copyto!(stokes.V.Vx, Vx0); copyto!(stokes.V.Vy, Vy0)
        Rx2, Ry2 = @zeros(nx - 1, ny), @zeros(nx, ny - 1)
        @parallel (@idx ni) JR2K.compute_Res!(
            Rx2, Ry2, @velocity(stokes)..., fields..., grid._di.center, grid._di.vertex, 0.4,
        )
        @test Array(Rx2) ≈ Rx_ref
        @test Array(Ry2) ≈ Ry_ref

        # ρgy linear in y: -∂y(ρg) dt Vy stabilization and the body force evaluated midway
        # between the two adjacent centers
        h0, h1, dtfs = 1.0, -0.6, 0.4
        ρgy_lin = [h0 + h1 * y for x in xc, y in yc]
        @parallel (@idx ni) JR2K.compute_Res!(
            Rx2, Ry2, @velocity(stokes)..., dev(Pf), dev(τxx), dev(τyy), dev(τxy), dev(ρgx), dev(ρgy_lin),
            grid._di.center, grid._di.vertex, dtfs,
        )
        Ry_lin = [
            syy[2] + sxy[1] - p[2] - (h0 + h1 * (yc[j] + yc[j + 1]) / 2) + Vy0[i + 1, j + 1] * h1 * dtfs
                for i in 1:nx, j in 1:(ny - 1)
        ]
        @test Array(Ry2) ≈ Ry_lin

        # velocity update V += R ηdτ / ητ with ητ averaged onto the face
        ητ = [1.0 + 0.2 * i + 0.1 * j^2 for i in 1:nx, j in 1:ny]
        ηdτ = 0.05
        ητx = [(ητ[i, j] + ητ[i + 1, j]) / 2 for i in 1:(nx - 1), j in 1:ny]
        ητy = [(ητ[i, j] + ητ[i, j + 1]) / 2 for i in 1:nx, j in 1:(ny - 1)]
        @parallel JR2K.compute_V!(
            @velocity(stokes)..., fields[1:4]..., ηdτ, fields[5:6]..., dev(ητ), grid._di.center, grid._di.vertex,
        )
        Vx_h, Vy_h = Array(stokes.V.Vx), Array(stokes.V.Vy)
        @test Vx_h[2:(end - 1), 2:(end - 1)] ≈ Vx0[2:(end - 1), 2:(end - 1)] .+ Rx_ref .* ηdτ ./ ητx
        @test Vy_h[2:(end - 1), 2:(end - 1)] ≈ Vy0[2:(end - 1), 2:(end - 1)] .+ Ry_ref .* ηdτ ./ ητy
        # untouched ghost/boundary rows
        @test Vx_h[:, 1] == Vx0[:, 1] && Vx_h[1, :] == Vx0[1, :]

        # free-surface stabilized update: vertical pseudo-time step ηdτ / (ητ - ηdτ dt ∂y(ρg))
        copyto!(stokes.V.Vx, Vx0); copyto!(stokes.V.Vy, Vy0)
        @parallel JR2K.compute_V!(
            @velocity(stokes)..., dev(Pf), dev(τxx), dev(τyy), dev(τxy), ηdτ, dev(ρgx), dev(ρgy_lin), dev(ητ),
            grid._di.center, grid._di.vertex, dtfs,
        )
        Vx_h, Vy_h = Array(stokes.V.Vx), Array(stokes.V.Vy)
        @test Vx_h[2:(end - 1), 2:(end - 1)] ≈ Vx0[2:(end - 1), 2:(end - 1)] .+ Rx_ref .* ηdτ ./ ητx
        @test Vy_h[2:(end - 1), 2:(end - 1)] ≈ Vy0[2:(end - 1), 2:(end - 1)] .+ Ry_lin .* ηdτ ./ (ητy .- ηdτ * dtfs * h1)
    end

    @testset "principal stresses" begin
        stokes = StokesArrays(backend, ni)
        σ = JR2.PrincipalStress(backend, ni)
        function check_principal(σ, τxx, τyy, τxy)
            σ1, σ2 = Array(σ.σ1), Array(σ.σ2)
            ok = true
            for j in 1:ny, i in 1:nx
                E = eigen(Symmetric([τxx[i, j] τxy[i, j]; τxy[i, j] τyy[i, j]]))
                # σ1 = λmax e_max, σ2 = λmin e_min, eigenvectors defined up to sign
                ok &= norm(σ1[:, i, j]) ≈ abs(E.values[2]) && abs(dot(σ1[:, i, j], E.vectors[:, 2])) ≈ abs(E.values[2])
                ok &= norm(σ2[:, i, j]) ≈ abs(E.values[1]) && abs(dot(σ2[:, i, j], E.vectors[:, 1])) ≈ abs(E.values[1])
            end
            return ok
        end
        # equal normal stresses: principal axes at ±45°, eigenvalues τxx ± |τxy|
        τn = [0.3 * sin(i + j) for i in 1:nx, j in 1:ny]
        τs = [0.5 + 0.2 * cos(i * j) for i in 1:nx, j in 1:ny]
        copyto!(stokes.τ.xx, τn); copyto!(stokes.τ.yy, τn); copyto!(stokes.τ.xy_c, τs)
        JR2.compute_principal_stresses!(stokes, σ)
        @test check_principal(σ, τn, τn, τs)
        # diagonal tensors: σ1 = max(τxx, τyy) along its own axis
        σd = JR2.PrincipalStress(backend, (2, 1))
        sd = StokesArrays(backend, (2, 1))
        copyto!(sd.τ.xx, [3.0, 1.0]); copyto!(sd.τ.yy, [1.0, 3.0]); copyto!(sd.τ.xy_c, [0.0, 0.0])
        JR2.compute_principal_stresses!(sd, σd)
        σ1d, σ2d = Array(σd.σ1), Array(σd.σ2)
        @test σ1d[:, 1, 1] ≈ [3.0, 0.0] && σ2d[:, 1, 1] ≈ [0.0, 1.0]
        @test abs.(σ1d[:, 2, 1]) ≈ [0.0, 3.0] && abs.(σ2d[:, 2, 1]) ≈ [1.0, 0.0]
        # general stress state
        copyto!(stokes.τ.xx, τxx_c); copyto!(stokes.τ.yy, τyy_c); copyto!(stokes.τ.xy_c, τxy_oc)
        σn = JR2.compute_principal_stresses(backend, stokes)
        @test check_principal(σn, τxx_c, τyy_c, τxy_oc)
    end

    @testset "viscous compute_τ!" begin
        θ = 0.6
        τxx, τyy, τxy = dev(τxx_c), dev(τyy_c), dev(τxy_v)
        @parallel (@idx ni .+ 1) JR2K.compute_τ!(τxx, τyy, τxy, dev(εxx_c), dev(εyy_c), dev(εxy_v), dev(η_c), θ)
        @test Array(τxx) ≈ @. (θ * τxx_c + 2η_c * εxx_c) / (θ + 1)
        @test Array(τyy) ≈ @. (θ * τyy_c + 2η_c * εyy_c) / (θ + 1)
        τxy_ref = [(θ * τxy_v[i, j] + 2vmean(η_c, i, j) * εxy_v[i, j]) / (θ + 1) for i in 1:(nx + 1), j in 1:(ny + 1)]
        @test Array(τxy) ≈ τxy_ref
    end
end

# ---------------------------------------------------------------------------------------
# Pseudo-transient solver: homogeneous pure shear, whose exact discrete solution is
# V = (εbg x, -εbg y), P = P0 and a spatially uniform stress, on any grid
# ---------------------------------------------------------------------------------------

const ni_s = (16, 12)
const εbg = 1.0

function pure_shear_state(grid)
    stokes = StokesArrays(backend, ni_s)
    xvx, yvx = Array.(grid.xi_vel[1])
    xvy, yvy = Array.(grid.xi_vel[2])
    # exact field on the boundary, perturbed in the interior so the solver has work to do
    Vx = [εbg * x for x in xvx, y in yvx]
    Vy = [-εbg * y for x in xvy, y in yvy]
    Vx[2:(end - 1), 2:(end - 1)] .+= [0.05 * sin(3x) * cos(2y) for x in xvx[2:(end - 1)], y in yvx[2:(end - 1)]]
    Vy[2:(end - 1), 2:(end - 1)] .+= [0.05 * cos(2x) * sin(3y) for x in xvy[2:(end - 1)], y in yvy[2:(end - 1)]]
    copyto!(stokes.V.Vx, Vx)
    copyto!(stokes.V.Vy, Vy)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    return stokes, flow_bcs
end

function check_pure_shear(stokes, grid, τxx_expected; rtol = 1.0e-6)
    xvx, yvx = Array.(grid.xi_vel[1])
    xvy, yvy = Array.(grid.xi_vel[2])
    Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
    @test Vx[:, 2:(end - 1)] ≈ [εbg * x for x in xvx, y in yvx[2:(end - 1)]] rtol = rtol
    @test Vy[2:(end - 1), :] ≈ [-εbg * y for x in xvy[2:(end - 1)], y in yvy] rtol = rtol
    @test all(isapprox.(Array(stokes.τ.xx), τxx_expected; rtol))
    @test all(isapprox.(Array(stokes.τ.yy), -τxx_expected; rtol))
    @test maximum(abs, Array(stokes.τ.xy)) < rtol * τxx_expected
    return nothing
end

solver_kwargs = (; iterMax = 20.0e3, nout = 100, verbose = false)
# the phase-ratio solver reports every convergence check regardless of `verbose`
quiet(f) = redirect_stdout(f, devnull)

@testset "Stokes solve! 2D" begin
    igg = IGG(init_global_grid(ni_s..., 1; init_MPI = JustRelax.MPI.Initialized() ? false : true, select_device = false, quiet = true)...)
    xv = stretched(ni_s[1], 1.0, 0.4, 0.2)
    yv = stretched(ni_s[2], 0.8, 0.3, 1.0; x0 = -0.8)
    grid_nu = Geometry(PTArray(backend), xv, yv)
    grid_u = Geometry(ni_s, (1.0, 0.8); origin = (0.0, -0.8))
    di_min = (minimum(diff(xv)), minimum(diff(yv)))
    pt_stokes = PTStokesCoeffs((1.0, 0.8), di_min; ϵ_rel = 1.0e-12, ϵ_abs = 1.0e-10, CFL = 0.95 / √2.1)
    ρg = @zeros(ni_s...), @zeros(ni_s...)
    η0, G0, K0, dt = 1.3, 1.0, 5.0, 0.8

    @testset "visco-elastic G, K form" begin
        stokes, flow_bcs = pure_shear_state(grid_nu)
        stokes.viscosity.η .= η0
        G = @fill(G0, ni_s...)
        K = @fill(K0, ni_s...)
        quiet(() -> solve!(stokes, pt_stokes, grid_nu, flow_bcs, ρg, G, K, dt, igg; kwargs = solver_kwargs))
        check_pure_shear(stokes, grid_nu, 2 * ηve(η0, G0, dt) * εbg)
        @test maximum(abs, Array(stokes.P)) < 1.0e-8
    end

    visc = LinearViscous(; η = η0)
    el = ConstantElasticity(; G = G0, Kb = K0)
    rheo_ve = SetMaterialParams(;
        Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
        CompositeRheology = CompositeRheology((visc, el)), Elasticity = el,
    )

    @testset "single MaterialParams form" begin
        stokes, flow_bcs = pure_shear_state(grid_nu)
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        quiet(() -> solve!(stokes, pt_stokes, grid_nu, flow_bcs, ρg, rheo_ve, args, dt, igg; kwargs = solver_kwargs))
        check_pure_shear(stokes, grid_nu, 2 * ηve(η0, G0, dt) * εbg)
    end

    @parallel_indices (I...) function fill_phase1!(phases)
        @index phases[1, I...] = 1.0
        return nothing
    end
    function single_phase_ratios()
        phase_ratios = PhaseRatios(backend_JP, 1, ni_s)
        @parallel (@idx ni_s) fill_phase1!(phase_ratios.center)
        @parallel (@idx ni_s .+ 1) fill_phase1!(phase_ratios.vertex)
        return phase_ratios
    end

    # strain-increment mode is exercised on the uniform grid only: it differentiates the
    # displacement normal components with the cell-center spacing (Stokes2D.jl)
    @testset "phase-ratio form, visco-elastic, strain_increment = $inc" for (inc, grid) in ((false, grid_nu), (true, grid_u))
        stokes, flow_bcs = pure_shear_state(grid)
        phase_ratios = single_phase_ratios()
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        compute_viscosity!(stokes, phase_ratios, args, (rheo_ve,), (-Inf, Inf))
        quiet() do
            solve!(
                stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, (rheo_ve,), args, dt, igg;
                kwargs = (; solver_kwargs..., strain_increment = inc),
            )
        end
        check_pure_shear(stokes, grid, 2 * ηve(η0, G0, dt) * εbg)
        @test maximum(abs, Array(stokes.P)) < 1.0e-8
    end

    @testset "phase-ratio form, regularised Drucker-Prager" begin
        C, ϕ, ηvp = 0.4, 30.0, 0.2
        pl = DruckerPrager_regularised(; C, ϕ, Ψ = 0.0, η_vp = ηvp)
        rheo_pl = SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el, pl)), Elasticity = el,
        )
        stokes, flow_bcs = pure_shear_state(grid_nu)
        phase_ratios = single_phase_ratios()
        args = (; T = @zeros(ni_s .+ 2...), P = stokes.P, dt)
        compute_viscosity!(stokes, phase_ratios, args, (rheo_pl,), (-Inf, Inf))
        quiet() do
            solve!(
                stokes, pt_stokes, grid_nu, flow_bcs, ρg, phase_ratios, (rheo_pl,), args, dt, igg;
                kwargs = solver_kwargs,
            )
        end
        # P = 0 and ψ = 0: τII = τy + η_vp λ with τII = τII_M - ηve λ, so the stress is the
        # ηve / η_vp weighted blend of the yield stress τy and the Maxwell stress τII_M
        ηe = ηve(η0, G0, dt)
        τy, τM = C * cosd(ϕ), 2ηe * εbg
        @test τM > τy
        τII = (ηe * τy + ηvp * τM) / (ηe + ηvp)
        check_pure_shear(stokes, grid_nu, τII)
        # accumulated plastic strain EII_pl = dt εII_pl = dt λ / 2
        λ = (τM - τy) / (ηe + ηvp)
        @test all(isapprox.(Array(stokes.EII_pl), dt * λ / 2; rtol = 1.0e-6))
    end
    @testset "grid-spacing call forms agree with the Geometry forms" begin
        # the `di` methods rebuild a uniform grid from the spacing; on a uniform grid both
        # call forms run the same iteration
        phase_ratios = single_phase_ratios()
        K, G = @fill(K0, ni_s...), @fill(G0, ni_s...)
        forms = (
            (s -> (s.viscosity.η .= η0; (G, K, dt)), false),
            (s -> (rheo_ve, (; T = @zeros(ni_s .+ 2...), P = s.P, dt), dt), false),
            (s -> (phase_ratios, (rheo_ve,), (; T = @zeros(ni_s .+ 2...), P = s.P, dt), dt), true),
        )
        for (form_args, phase_form) in forms
            results = map((grid_u, grid_u.di, grid_u.di.center)) do g
                stokes, flow_bcs = pure_shear_state(grid_u)
                fargs = form_args(stokes)
                phase_form && compute_viscosity!(stokes, fargs[1], fargs[3], fargs[2], (-Inf, Inf))
                quiet(() -> solve!(stokes, pt_stokes, g, flow_bcs, ρg, fargs..., igg; kwargs = (; solver_kwargs..., verbose = true)))
                Array.((stokes.V.Vx, stokes.V.Vy, stokes.τ.xx, stokes.τ.xy))
            end
            @test all(results[2] .≈ results[1])
            @test all(results[3] .≈ results[1])
            @test all(isapprox.(results[1][3], 2 * ηve(η0, G0, dt) * εbg; rtol = 1.0e-6))
        end
    end

    finalize_global_grid(; finalize_MPI = true)
end
