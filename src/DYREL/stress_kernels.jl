function compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation, dt)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy_c), # centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    ## VERTEX CALCULATION
    @inbounds begin
        Ic = clamped_indices(ni, I...)
        τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
        εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
        λvij = λv[I...]
        # ηij   = harm_clamped(η, Ic...)
        ηij = ηv[I...]
        Pij = av_clamped(P, Ic...)
        ratio = phase_ratios_vertex[I...]
        # compute local stress
        τxx_I, τyy_I, τxy_I, _, _, _, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt)

        # update arrays
        τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
        λv[I...] = λ_I

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
            εij = ε[1][I...], ε[2][I...], av(ε[3])
            λij = λ[I...]
            ηij = η[I...]
            Pij = P[I...]
            ratio = phase_ratios_center[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I = compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt)
            # update arrays
            τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = εxx_pl, εyy_pl, εxy_pl
            τII[I...] = τII_I
            η_vep[I...] = ηvep_I
            λ[I...] = λ_I
            ΔPψ[I...] = ΔPψ_I
        end
    end

    return nothing
end

@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt) where {N}
    return quote
        @inline
        # iterate over phases
        v_phases = Base.@ntuple $N phase -> begin
            # get phase ratio
            ratio_I = phase_ratio[phase]
            v = if iszero(ratio_I) # this phase does not contribute
                empty_stress_solution(εij)

            else
                # get rheological properties for this phase
                G = get_shear_modulus(rheology, phase)
                Kb = get_bulk_modulus(rheology, phase)
                ispl, C, sinϕ, cosϕ, sinΨ, η_reg = plastic_params(rheology[phase].CompositeRheology[1].elements, 0.0e0) # this 0e0 is accumulated plastic strain, not used here
                # compute local stress
                ratio_I .* _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, dt)
            end
        end
        # sum contributions from all phases
        v = reduce(.+, v_phases)
        return v # this returns (τ_ij...), (εij_pl...), τII, λ, ΔPψ, ηvep
    end
end

@inline function _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, dt)

    # viscoelastic viscosity
    η_ve = isinf(G) ?
        inv(inv(η) + inv(G * dt)) :
        (η * G * dt) / (η + G * dt) # more efficient than inv(inv(η) + inv(G * dt))
    # effective strain rate
    inv_2Gdt = inv(2 * G * dt)
    εij_eff = @. εij + τij_o * inv_2Gdt

    εII = second_invariant(εij_eff)

    # early return if there is no deformation
    iszero(εII) && return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η)

    # Plastic stress correction starts here
    τij = @. 2 * η_ve * εij_eff
    τII = second_invariant(τij)
    # F = τII - C * cosϕ - max(P, 0.0e0) * sinϕ
    F = τII - C * cosϕ - P * sinϕ
    # F = if P ≥ 0
    #     # DP in extension
    #     τII - C * cosϕ - P * sinϕ
    # else
    #     # VM in extension
    #     τII - C
    # end
    λ = if ispl && F ≥ 0
        λ_new = F / (η_ve + η_reg + Kb * dt * sinϕ * sinΨ)
        λ_relaxation * λ_new + (1 - λ_relaxation) * λ
    else
        0.0
    end
    # Effective viscoelastic-plastic viscosity
    η_vep = (τII - λ * η_ve) / (2 * εII)
    # Update stress and plastic strain rate
    τij, τII, εij_pl, ΔPψ = if λ > 0
        τij = @. 2 * η_vep * εij_eff
        τII = second_invariant(τij)
        εij_pl = @. λ * 0.5 * τij / τII # λ * dQdτij
        # Update pressure correction due to dilatation
        ΔPψ = iszero(sinΨ) ? 0.0 : λ * sinΨ * Kb * dt
        τij, τII, εij_pl, ΔPψ
    else
        εij_pl = zero_tuple(εij)
        ΔPψ = 0.0
        τij, τII, εij_pl, ΔPψ
    end

    return τij..., εij_pl..., τII, λ, ΔPψ, η_vep
end

# this returns zero for: τxx, τyy, τxy, τII, ηvep, λ, ΔPψ
@inline empty_stress_solution(::NTuple{3, T}) where {T} = zero_tuple(T, Val(10))
# this returns zero for: τxx, τyy, τyy, τyz, τxz, τxy, τII, ηvep, λ, ΔPψ
@inline empty_stress_solution(::NTuple{6, T}) where {T} = zero_tuple(T, Val(13))

@inline zero_tuple(::Type{T}, ::Val{N}) where {T, N} = ntuple(_ -> zero(T), Val(N))
@inline zero_tuple(::NTuple{N, T}) where {T, N} = zero_tuple(T, Val(N))


## VARIATIONAL STOKES STRESS KERNELS

function compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy_c), # centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt
    )
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(
        τ,
        τ_v,
        τ_o,
        τ_ov,
        τII,
        ε,
        ε_pl,
        P,
        λ,
        λv,
        η,
        η_vep,
        ΔPψ,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt
    )

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    @inbounds begin
        ## VERTEX CALCULATION
        @inbounds if isvalid_v(ϕ, I...)
            Ic = clamped_indices(ni, I...)
            τij_o = τ_ov[1][I...], τ_ov[2][I...], τ_ov[3][I...]
            εij = av_clamped(ε[1], Ic...), av_clamped(ε[2], Ic...), ε[3][I...]
            λvij = λv[I...]
            ηij = harm_clamped(η, Ic...)
            Pij = av_clamped(P, Ic...)
            ratio = phase_ratios_vertex[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, _, _, _, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt)

            # update arrays
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
            λv[I...] = λ_I

        else
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = 0.0e0, 0.0e0, 0.0e0
            λv[I...] = 0.0e0

        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            @inbounds if isvalid_c(ϕ, I...)
                τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
                εij = ε[1][I...], ε[2][I...], av(ε[3])
                λij = λ[I...]
                ηij = η[I...]
                Pij = P[I...]
                ratio = phase_ratios_center[I...]

                # compute local stress
                τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I = compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt)
                # update arrays
                τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = εxx_pl, εyy_pl, εxy_pl
                τII[I...] = τII_I
                η_vep[I...] = ηvep_I
                λ[I...] = λ_I
                ΔPψ[I...] = ΔPψ_I

            else
                τ[1][I...], τ[2][I...], τ[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = 0.0e0, 0.0e0, 0.0e0
                τII[I...] = 0.0e0
                η_vep[I...] = 0.0e0
                λ[I...] = 0.0e0
                ΔPψ[I...] = 0.0e0

            end
        end
    end

    return nothing
end
