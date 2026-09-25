pushfirst!(LOAD_PATH, dirname(@__DIR__))
using Test, JustRelax, GeoParams, StaticArrays, ForwardDiff

const JRCap = JustRelax.JustRelax2D

function cap_material(; Ψ = 0.0, η_vp = 0.1)
    pl = DruckerPragerCap(; C = 1.0, ϕ = 30.0, Ψ, η_vp, pT = -0.5)
    el = ConstantElasticity(; G = 1.0, Kb = 4.0)
    material = SetMaterialParams(;
        Phase = 1, Elasticity = el,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, pl)),
    )
    return material, pl
end

@testset "Coupled tensile-cap return map" begin
    material, pl = cap_material()
    rheology = (material,)
    @testset "Hydrostatic tension and mixed loading" begin
        for JR in (JustRelax.JustRelax2D, JustRelax.JustRelax3D)
            N = JR === JustRelax.JustRelax2D ? 3 : 6
            for shear in (0.0, 1.0), P in (-1.0, -0.8)
                strain = ntuple(i -> i == N ? shear : 0.0, N)
                old = ntuple(_ -> 0.0, N)
                out = @inferred JR._compute_local_stress(strain, old, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
                s, λ, dp, _, volume = out[(2N + 1):end]
                p = P + dp
                Aτ = GeoParams.∂Q∂τII(pl, s; P = p)
                Ap = -GeoParams.∂Q∂P(pl, p; τII = s)
                @test λ > 0
                @test s - shear + λ * Aτ ≈ 0 atol = 1.0e-12
                @test dp - 4λ * Ap ≈ 0 atol = 1.0e-12
                @test GeoParams.compute_yieldfunction(pl; P = p, τII = s) - 0.1λ ≈ 0 atol = 1.0e-12
                @test volume ≈ λ * Ap
                if iszero(shear)
                    a = sqrt(1 + sind(30.0)^2)
                    expected_λ = a * (-0.5 - P) / (0.1 + 4a)
                    @test λ ≈ expected_λ
                    @test all(iszero, out[1:N])
                    @test volume > 0
                end
            end
        end
    end
    @testset "Elastic unloading and phase selection" begin
        τ = (0.0, 0.0, 0.01)
        for phase in (1, (1.0,), SVector(1.0))
            λ, _, _ = @inferred JRCap.plastic_correction(rheology, phase, τ, 1.0, 0.0, 0.5, 4.0, 0.1, 2.0, 0.2, true)
            @test iszero(λ)
            x, ok = @inferred JRCap.cap_return_mapping(rheology, phase, 1.0, -0.8, 0.0, 0.5, 4.0, 0.1)
            @test ok
            @test x isa SVector{3, Float64}
        end
    end
    @testset "AD Jacobian and state forwarding" begin
        x = SVector(0.4, -0.3, 0.2)
        f = y -> JRCap.cap_residual(y, rheology, 1, 0.0, SVector(1.0, -0.8), 0.5, 4.0, 0.1, 0.0)
        J = @inferred ForwardDiff.jacobian(f, x)
        @test J isa SMatrix{3, 3, Float64, 9}
        h = 1.0e-6
        for i in 1:3
            d = SVector(ntuple(j -> i == j ? h : 0.0, 3))
            @test J[:, i] ≈ (f(x + d) - f(x - d)) / (2h) rtol = 1.0e-7 atol = 1.0e-9
        end
        g, Qp, _ = JRCap.compute_plastic_gradients_phase(rheology, 1, (0.0, 0.0, 1.0); P = -0.8, τII = 1.0, EII = 0.0)
        @test g[3] ≈ GeoParams.∂Q∂τII(pl, 1.0; P = -0.8)
        @test Qp ≈ GeoParams.∂Q∂P(pl, -0.8; τII = 1.0)
    end
    @testset "Failure is not an admissible stress" begin
        _, ok = JRCap.cap_return_mapping(rheology, 1, 1.0, -0.8, 0.0, 0.5, 4.0, 0.1; maxiter = 0)
        @test !ok
        λ, _, _ = JRCap.plastic_correction(rheology, 1, (0.0, 0.0, 1.0), -0.8, 0.0, 0.5, Inf, 0.1, 0.0, 1.0, true)
        @test isnan(λ)
    end
end

function dp_material(; Ψ = 0.0, η_vp = 0.1, G = 1.0, Kb = 4.0)
    pl = DruckerPrager_regularised(; C = 1.0, ϕ = 30.0, Ψ, η_vp)
    el = ConstantElasticity(; G, Kb)
    material = SetMaterialParams(;
        Phase = 1, Elasticity = el,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, pl)),
    )
    return material, pl
end

function cap_material_ve(; Ψ = 0.0, η_vp = 0.1, G = 1.0, Kb = 4.0, pT = -0.5)
    pl = DruckerPragerCap(; C = 1.0, ϕ = 30.0, Ψ, η_vp, pT)
    el = ConstantElasticity(; G, Kb)
    material = SetMaterialParams(;
        Phase = 1, Elasticity = el,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), el, pl)),
    )
    return material, pl
end

# Residuals of Eq. (42) evaluated at the state the local solve returned.
function cap_residuals(pl, strain, P, out, N; η = 1.0, G = 1.0, Kb = 4.0, dt = 1.0, EII = 0.0)
    s, λ, dp, _, volume = out[(2N + 1):end]
    η_ve = η / (1 + η / (G * dt))
    s_trial = 2 * η_ve * second_invariant(ntuple(i -> strain[i], N))
    p = P + dp
    Aτ = GeoParams.∂Q∂τII(pl, s; P = p, EII = EII)
    Ap = -GeoParams.∂Q∂P(pl, p; τII = s, EII = EII)
    F = GeoParams.compute_yieldfunction(pl; P = p, τII = s, EII = EII)
    return (
        s = s, λ = λ, dp = dp, volume = volume,
        R_s = s - s_trial + 2 * η_ve * λ * Aτ,
        R_p = dp - Kb * dt * λ * Ap,
        R_f = F - pl.η_vp.val * λ, F = F,
    )
end

@testset "Cap return map: regimes, limits and invariance" begin

    @testset "Admissible elastic trial and unloading" begin
        material, _ = cap_material_ve()
        for JR in (JustRelax.JustRelax2D, JustRelax.JustRelax3D)
            N = JR === JustRelax.JustRelax2D ? 3 : 6
            strain = ntuple(i -> i == N ? 0.01 : 0.0, N)
            old = ntuple(_ -> 0.0, N)
            for λ_old in (0.0, 0.7)
                out = JR._compute_local_stress(strain, old, 1.0, 5.0, 1.0, 4.0, λ_old, 1.0, material, 1.0, 0.0)
                s, λ, dp, _, volume = out[(2N + 1):end]
                @test iszero(λ)                  # stale multiplier is cleared on unloading
                @test iszero(dp)
                @test iszero(volume)
                @test s ≈ 0.01                   # 2*η_ve = 1 here, so τII equals the trial
                @test out[N] ≈ 0.01
            end
        end
    end

    @testset "Drucker-Prager limit away from the cap" begin
        cap, _ = cap_material_ve()
        dp_mat, _ = dp_material()
        for JR in (JustRelax.JustRelax2D, JustRelax.JustRelax3D)
            N = JR === JustRelax.JustRelax2D ? 3 : 6
            strain = ntuple(i -> i == N ? 4.0 : 0.0, N)
            old = ntuple(_ -> 0.0, N)
            out_cap = JR._compute_local_stress(strain, old, 1.0, 2.0, 1.0, 4.0, 0.0, 1.0, cap, 1.0, 0.0)
            out_dp = JR._compute_local_stress(strain, old, 1.0, 2.0, 1.0, 4.0, 0.0, 1.0, dp_mat, 1.0, 0.0)
            # Compressive state: the cap branch is inactive, so the converged
            # Newton solution must match the analytical cone correction.
            @test out_cap[2N + 2] > 0   # the comparison is only meaningful while yielding
            for k in 1:(2N + 5)
                @test out_cap[k] ≈ out_dp[k] atol = 1.0e-10
            end
        end
    end

    @testset "Regularization, timestep, bulk modulus and history" begin
        for η_vp in (0.0, 0.1, 1.0), dt in (0.1, 1.0, 10.0), Kb in (4.0, 1.0e6), EII in (0.0, 0.3)
            material, pl = cap_material_ve(; η_vp, Kb, Ψ = 5.0)
            for shear in (0.0, 1.0), P in (-1.0, -0.8, 0.5)
                N = 3
                strain = ntuple(i -> i == N ? shear : 0.0, N)
                old = ntuple(_ -> 0.0, N)
                out = JustRelax.JustRelax2D._compute_local_stress(
                    strain, old, 1.0, P, 1.0, Kb, 0.0, 1.0, material, dt, EII
                )
                r = cap_residuals(pl, strain, P, out, N; Kb, dt, EII)
                @test r.λ ≥ 0
                @test r.s ≥ 0
                @test abs(r.R_s) < 1.0e-9
                @test abs(r.R_p) < 1.0e-9 * max(1, Kb * dt)
                # complementarity: the regularized yield equation holds only when yielding
                if r.λ > 0
                    @test abs(r.R_f) < 1.0e-9
                else
                    @test r.F <= 1.0e-12
                end
                @test r.volume ≈ r.λ * (-GeoParams.∂Q∂P(pl, P + r.dp; τII = r.s, EII = EII))
                # Dilatant flow: yielding pushes the pressure away from tension.
                r.λ > 0 && @test r.dp ≥ 0
            end
        end
    end

    @testset "Fluid pressure enters as the effective pressure P - Pf" begin
        cap, _ = cap_material_ve(; Ψ = 5.0)
        dp_mat, _ = dp_material()
        for JR in (JustRelax.JustRelax2D, JustRelax.JustRelax3D), material in (cap, dp_mat)
            N = JR === JustRelax.JustRelax2D ? 3 : 6
            old = ntuple(_ -> 0.0, N)
            for shear in (0.0, 1.0, 4.0), (P, Pf) in ((0.5, 1.3), (2.0, 1.5), (2.0, 0.0))
                strain = ntuple(i -> i == N ? shear : 0.0, N)
                wet = JR._compute_local_stress(strain, old, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0, Pf)
                dry = JR._compute_local_stress(strain, old, 1.0, P - Pf, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
                for k in eachindex(wet, dry)
                    @test wet[k] ≈ dry[k] atol = 1.0e-10
                end
            end
            # Pf lowers the effective pressure enough to turn an elastic state into a yielding one
            strain = ntuple(i -> i == N ? 1.0 : 0.0, N)
            @test iszero(JR._compute_local_stress(strain, old, 1.0, 3.0, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)[2N + 2])
            @test JR._compute_local_stress(strain, old, 1.0, 3.0, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0, 3.0)[2N + 2] > 0
        end
    end

    @testset "Invariance under rotation of the strain tensor" begin
        material, _ = cap_material_ve()
        old2 = ntuple(_ -> 0.0, 3)
        old3 = ntuple(_ -> 0.0, 6)
        for P in (-1.0, -0.8, 0.5)
            # (a, -a, 0) and (0, 0, a) are the same deviatoric state rotated by 45 degrees
            shear = JustRelax.JustRelax2D._compute_local_stress((0.0, 0.0, 1.0), old2, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
            pure = JustRelax.JustRelax2D._compute_local_stress((1.0, -1.0, 0.0), old2, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
            for k in 7:11   # τII, λ, ΔPψ, η_vep, ε_vol_pl
                @test shear[k] ≈ pure[k] atol = 1.0e-12
            end
            # the same state embedded in 3D, on two different shear slots
            yz = JustRelax.JustRelax3D._compute_local_stress((0.0, 0.0, 0.0, 1.0, 0.0, 0.0), old3, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
            xy = JustRelax.JustRelax3D._compute_local_stress((0.0, 0.0, 0.0, 0.0, 0.0, 1.0), old3, 1.0, P, 1.0, 4.0, 0.0, 1.0, material, 1.0, 0.0)
            for k in 13:17
                @test yz[k] ≈ xy[k] atol = 1.0e-12
            end
        end
    end
end
