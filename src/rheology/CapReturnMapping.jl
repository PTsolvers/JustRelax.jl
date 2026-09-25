# Popov et al. (2025), Eq. 42: local coupled stress/pressure/Perzyna solve.
# Scalars and StaticArrays only: this routine also runs inside GPU kernels.
@inline has_tensile_cap(elements::Tuple) = any(v -> v isa DruckerPragerCap, elements)
@inline has_tensile_cap(rheology, phase::Integer) =
    has_tensile_cap(rheology[phase].CompositeRheology[1].elements)
@inline has_tensile_cap(rheology, ratio) =
    any(map((r, w) -> !iszero(w) && has_tensile_cap(r.CompositeRheology[1].elements), rheology, Tuple(ratio)))

@inline rheology_of(args::Tuple) = _rheology_of(args...)
@inline _rheology_of(rheology::NTuple{N, AbstractMaterialParamsStruct}, rest...) where {N} = rheology
@inline _rheology_of(rheology::AbstractMaterialParamsStruct, rest...) = (rheology,)
@inline _rheology_of(_, rest...) = _rheology_of(rest...)
@inline _rheology_of() = nothing

"""
    reject_incompressible_cap(rheology, solver)

Throw when a phase combines `DruckerPragerCap` with an infinite bulk modulus, naming the
`solver` that cannot integrate it.

Tensile cap flow is volumetric, so the local return map solves the pressure through the
elastic volumetric compliance `1/(K*dt)`. That compliance vanishes for an incompressible
phase (`ν = 0.5`, `Kb = Inf`), which leaves the corrected pressure undetermined. Without
this check the kernel reports a failed local solve and the run stops on a bare `NaN(s)`.
"""
function reject_incompressible_cap(rheology, solver)
    rheology === nothing && return nothing
    for r in rheology
        if has_tensile_cap(r.CompositeRheology[1].elements) && !isfinite(get_bulk_modulus(r))
            error(
                "$solver: phase $(r.Phase) combines `DruckerPragerCap` with an infinite bulk " *
                    "modulus (ν = 0.5). Tensile cap plasticity corrects the pressure through the " *
                    "elastic volumetric compliance and needs a finite `Kb`. Use a compressible " *
                    "elasticity for this phase, or a plasticity without a tensile cap."
            )
        end
    end
    return nothing
end

# GeoParams' invariant derivative returns Aτ = (∂Q/∂τII)/2. Using the scalar
# interface also avoids the GeoParams tensor wrappers dropping pressure kwargs
# (JuliaGeodynamics/GeoParams.jl#348).
@inline function cap_invariants(v::AbstractPlasticity, s, p, EII)
    args = (; P = p, τII = s, EII, Pf = zero(EII), perturbation_C = one(EII))
    return SVector(
        GeoParams.compute_yieldfunction(v; args...),
        GeoParams.∂Q∂τII(v, s; args...),
        -GeoParams.∂Q∂P(v, p; args...),
    )
end

# Adapter for the older cell-centered stress path.
@inline function update_cap_stress!(τ, τII, τ_old, ε, ε_pl, η_vep, λ, rheology, phase, P, EII, η, dτ_r, _Gdt, Kdt, ηvp, I...)
    τij, τold, εij = cache_tensors(τ, τ_old, ε, I...)
    dτij, _ = compute_stress_increment_and_trial(τij, τold, η, εij, _Gdt, dτ_r)
    trial = τij .+ dτij
    λ[I...], g, Qp = plastic_correction(rheology, phase, trial, P, EII, η * dτ_r, Kdt, ηvp, λ[I...], one(P), true)
    rate = λ[I...] .* g
    corrected = trial .- (2 * η * dτ_r) .* rate
    correct_stress!(τ, corrected, I...)
    update_plastic_strain_rate!(ε_pl, rate, I)
    τII[I...] = second_invariant(corrected)
    η_vep[I...] = effective_viscosity(τII[I...], second_invariant(εij), η)
    volume_rate = -λ[I...] * Qp
    return P + Kdt * volume_rate, volume_rate
end

@generated function cap_invariants(elements::Tuple, s, p, EII)
    N = length(elements.parameters)
    return quote
        Base.@inline
        Base.@nexprs $N i -> begin
            v = elements[i]
            isplastic(v) && return cap_invariants(v, s, p, EII)
        end
        # Match the existing phase-weighted yield/flow convention.
        return SVector(s, zero(s), zero(s))
    end
end

@inline cap_invariants(rheology, phase::Integer, s, p, EII) =
    cap_invariants(rheology[phase].CompositeRheology[1].elements, s, p, EII)

@inline function cap_invariants(rheology, ratio, s, p, EII)
    values = map(rheology, Tuple(ratio)) do r, w
        iszero(w) && return SVector(zero(s), zero(s), zero(s))
        return w * cap_invariants(r.CompositeRheology[1].elements, s, p, EII)
    end
    return reduce(+, values)
end

@inline function cap_residual(x, rheology, phase, EII, trial, η, Kdt, ηvp)
    s, p, λ = x
    F, Aτ, Ap = cap_invariants(rheology, phase, s, p, EII)
    return SVector(s - trial[1] + 2 * η * λ * Aτ, p - trial[2] - Kdt * λ * Ap, F - ηvp * λ)
end

function cap_return_mapping(rheology, phase, s::T, p::T, EII, η, Kdt, ηvp; maxiter = 40) where {T}
    F, _, _ = cap_invariants(rheology, phase, s, p, EII)
    x = SVector(s, p, zero(T))
    F <= 0 && return x, true
    # Dilatant cap flow needs a finite elastic volumetric compliance.
    isfinite(Kdt) && Kdt > 0 && η > 0 || return x, false
    scale = max(abs(s), abs(p), abs(F), eps(T))
    tolerance = 100 * eps(T)
    trial = SVector(s, p)
    residual = y -> cap_residual(y, rheology, phase, EII, trial, η, Kdt, ηvp) / scale
    r = residual(x)
    for _ in 1:maxiter
        maximum(abs, r) <= tolerance && return x, true
        J = ForwardDiff.jacobian(residual, x)
        step = J \ r
        all(isfinite, step) || return x, false
        α = one(T)
        accepted = false
        for _ in 1:24
            candidate = x - α * step
            if candidate[1] >= 0 && candidate[3] >= 0
                next_r = residual(candidate)
                if all(isfinite, next_r) && sum(abs2, next_r) <= (1 - T(1.0e-4) * α) * sum(abs2, r)
                    x, r = candidate, next_r
                    accepted = true
                    break
                end
            end
            α /= 2
        end
        accepted || return x, false
    end
    return x, maximum(abs, r) <= tolerance
end

# Shared by PT (η = viscosity*dτ_r) and DYREL (η = Maxwell viscosity).
# Keep the analytical, relaxed Drucker-Prager correction for non-cap materials.
@inline function plastic_correction(rheology, phase, τtrial::NTuple{N, T}, P, EII, η, Kdt, ηvp, λold, relλ, is_pl) where {N, T}
    s = second_invariant(τtrial)
    if has_tensile_cap(rheology, phase)
        x, converged = cap_return_mapping(rheology, phase, s, P, EII, η, Kdt, ηvp)
        _, Aτ, Ap = cap_invariants(rheology, phase, x[1], x[2], EII)
        direction = iszero(s) ? zero(T) : Aτ / s
        g = ntuple(i -> direction * τtrial[i], Val(N))
        # NaNs propagate through the stress/pressure residual to the existing
        # host-side convergence checks, rather than accepting a failed return map.
        # The converged multiplier is returned unrelaxed on purpose: relaxing it the way
        # the analytical branch does leaves the converged fields unchanged but costs PT
        # iterations (+15% at relλ = 0.2, +75% at 0.05 on the DPCap shear band).
        return converged ? x[3] : T(NaN), g, -Ap
    end
    g, Qp, Fp = compute_plastic_gradients_phase(rheology, phase, τtrial; P, τII = s, EII)
    F = compute_yieldfunction_phase(rheology, phase; P, τII = s, EII)
    volume = isinf(Kdt) ? zero(T) : Kdt * Fp * Qp
    λ = if is_pl && !iszero(s) && F > 0
        (1 - relλ) * λold + relλ * F / (η + ηvp + volume)
    else
        zero(T)
    end
    return λ, g, Qp
end
