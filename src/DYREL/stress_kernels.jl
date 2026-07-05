function compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, λ_relaxation, dt, do_partials = false)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy),   # centers
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        dyrel,
        dyrel.∂ΔPψc_∂ε,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt, do_partials
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
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        ηv,
        η_vep,
        ΔPψ,
        dyrel,
        ∂ΔPψc_∂ε,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt, do_partials::Bool
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
        EIIv = av_clamped(EII_pl, Ic...)
        ratio = phase_ratios_vertex[I...]
        # compute local stress
        τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv)

        # update arrays
        τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
        ε_pl[3][I...] = εxy_pl
        λv[I...] = λ_I

        if do_partials
            Jτ_vertex = local_stress_jacobian_ε(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIv)
            store_stress_jacobian_vertex!(dyrel, Jτ_vertex, I...)
        end

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = τ_o[1][I...], τ_o[2][I...], τ_o[3][I...]
            εij = ε[1][I...], ε[2][I...], av(ε[3])
            λij = λ[I...]
            ηij = η[I...]
            Pij = P[I...]
            EII = EII_pl[I...]
            ratio = phase_ratios_center[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
                compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EII)
            # update arrays
            τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[1][I...], ε_pl[2][I...] = εxx_pl, εyy_pl
            ε_vol_pl[I...] = ε_vol_pl_I
            τII[I...] = τII_I
            η_vep[I...] = ηvep_I
            λ[I...] = λ_I
            ΔPψ[I...] = ΔPψ_I

            if do_partials
                Jτ_center = local_stress_jacobian_ε(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EII)
                store_stress_jacobian_center!(dyrel, Jτ_center, I...)
                store_pressure_correction_jacobian!(∂ΔPψc_∂ε, Jτ_center, I...)
            end
        end
    end

    return nothing
end

@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt, EII) where {N}
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
                ratio_I .* _compute_local_stress(
                    εij, τij_o, η, P, G, Kb, λ, λ_relaxation, rheology[phase], dt, EII
                )
            end
        end
        # sum contributions from all phases
        v = reduce(.+, v_phases)
        return v # this returns (τ_ij...), (εij_pl...), τII, λ, ΔPψ, ηvep
    end
end

@inline function local_stress_jacobian_ε(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt, EII)
    εij_vec = SA[εij...]
    return ForwardDiff.jacobian(
        ε -> compute_local_stress(Tuple(ε), τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio, dt, EII),
        εij_vec,
    )
end

@inline function store_stress_jacobian_center!(dyrel, Jτ, I...)
    dyrel.∂τxxc_∂εxx[I...] = Jτ[1, 1]
    dyrel.∂τxxc_∂εyy[I...] = Jτ[1, 2]
    dyrel.∂τxxc_∂εxy[I...] = Jτ[1, 3]
    dyrel.∂τyyc_∂εxx[I...] = Jτ[2, 1]
    dyrel.∂τyyc_∂εyy[I...] = Jτ[2, 2]
    dyrel.∂τyyc_∂εxy[I...] = Jτ[2, 3]
    dyrel.∂τxyc_∂εxx[I...] = Jτ[3, 1]
    dyrel.∂τxyc_∂εyy[I...] = Jτ[3, 2]
    dyrel.∂τxyc_∂εxy[I...] = Jτ[3, 3]
    return nothing
end

@inline function store_stress_jacobian_vertex!(dyrel, Jτ, I...)
    dyrel.∂τxxv_∂εxx[I...] = Jτ[1, 1]
    dyrel.∂τxxv_∂εyy[I...] = Jτ[1, 2]
    dyrel.∂τxxv_∂εxy[I...] = Jτ[1, 3]
    dyrel.∂τyyv_∂εxx[I...] = Jτ[2, 1]
    dyrel.∂τyyv_∂εyy[I...] = Jτ[2, 2]
    dyrel.∂τyyv_∂εxy[I...] = Jτ[2, 3]
    dyrel.∂τxyv_∂εxx[I...] = Jτ[3, 1]
    dyrel.∂τxyv_∂εyy[I...] = Jτ[3, 2]
    dyrel.∂τxyv_∂εxy[I...] = Jτ[3, 3]
    return nothing
end

@inline function store_pressure_correction_jacobian!(∂ΔPψ_∂ε, Jτ, I...)
    ∂ΔPψ_∂ε[1][I...] = Jτ[9, 1]
    ∂ΔPψ_∂ε[2][I...] = Jτ[9, 2]
    ∂ΔPψ_∂ε[3][I...] = Jτ[9, 3]
    return nothing
end

@inline function _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, rheology, dt, EII)
    # Only is_pl and η_reg are still needed locally; F + gradients come from GeoParams.
    # EII=0 here matches the existing DYREL pattern (no EII-softening in the inner loop).
    ispl, _, _, _, _, η_reg = plastic_params(rheology, EII)

    # viscoelastic viscosity
    η_ve = isinf(G) ?
        inv(inv(η) + inv(G * dt)) :
        (η * G * dt) / (η + G * dt) # more efficient than inv(inv(η) + inv(G * dt))
    # effective strain rate
    inv_2Gdt = inv(2 * G * dt)
    εij_eff = @. εij + τij_o * inv_2Gdt

    εII = second_invariant(εij_eff)

    # early return if there is no deformation
    iszero(εII) && return SA[zero_tuple(εij)..., zero_tuple(εij)..., zero(εII), zero(εII), zero(εII), η, zero(εII)]

    # trial stress
    τij = @. 2 * η_ve * εij_eff
    τII = second_invariant(τij)

    # F + gradients at trial stress via GeoParams (DP cone, DPCap cone+cap, ...)
    elements = rheology.CompositeRheology[1].elements
    args = (; P = P, τII = τII, EII = EII)
    F = _yieldfunction_elements(elements, args)
    dQdτ, dQdP, dFdP = _plastic_grad_elements(elements, τij, args)
    # dQdτ is already in tensor convention (shear slots halved inside _plastic_grad_primitive)

    λ, ε_vol_pl = if ispl && F ≥ 0
        λ_new = F / (η_ve + η_reg + Kb * dt * dFdP * dQdP)
        λ_relaxation * λ_new + (1 - λ_relaxation) * λ
        # Volumetric plastic strain rate
        ε_vol_pl = -λ * dQdP
        λ_new, ε_vol_pl
    else
        0.0, 0.0
    end

    # Update stress and plastic strain rate
    τij, τII, εij_pl, ΔPψ = if λ > 0
        εij_pl = @. λ * dQdτ
        τij = @. τij - 2.0 * η_ve * εij_pl
        τII = second_invariant(τij)
        # Pressure correction from volumetric flow
        ΔPψ = iszero(dQdP) ? 0.0 : -λ * dQdP * Kb * dt
        τij, τII, εij_pl, ΔPψ
    else
        εij_pl = zero_tuple(εij)
        ΔPψ = 0.0
        τij, τII, εij_pl, ΔPψ
    end

    # Effective viscoelastic-plastic viscosity
    η_vep = τII * 0.5 * inv(second_invariant(εij))

    return SA[τij..., εij_pl..., τII, λ, ΔPψ, η_vep, ε_vol_pl]
end

# this returns zero for: τxx, τyy, τxy, εxx_pl, εyy_pl, εxy_pl, τII, λ, ΔPψ, ηvep, ε_vol_pl
@inline empty_stress_solution(::NTuple{3, T}) where {T} = zero_static_vector(T, Val(11))
# 3D placeholder: τxx, τyy, τzz, τyz, τxz, τxy, εxx_pl..εxy_pl, τII, λ, ΔPψ, ηvep, ε_vol_pl
@inline empty_stress_solution(::NTuple{6, T}) where {T} = zero_static_vector(T, Val(17))

@inline zero_tuple(::Type{T}, ::Val{N}) where {T, N} = ntuple(_ -> zero(T), Val(N))
@inline zero_tuple(::NTuple{N, T}) where {T, N} = zero_tuple(T, Val(N))
@inline zero_static_vector(::Type{T}, ::Val{N}) where {T, N} = SA[zero_tuple(T, Val(N))...]


## VARIATIONAL STOKES STRESS KERNELS

function compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt, do_partials = false)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),          # centers
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),        # vertices
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),    # centers
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),  # vertices
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),            # staggered grid
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy), # centers
        stokes.EII_pl,                                      # accumulated plastic strain rate @ centers
        stokes.ε_vol_pl,                                    # volumetric plastic strain rate @ centers
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        dyrel,
        dyrel.∂ΔPψc_∂ε,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt, do_partials
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
        EII_pl,
        ε_vol_pl,
        P,
        λ,
        λv,
        η,
        η_vep,
        ΔPψ,
        dyrel,
        ∂ΔPψc_∂ε,
        ϕ::JustRelax.RockRatio,
        rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt, do_partials::Bool
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
            EIIvij = av_clamped(EII_pl, Ic...)
            ratio = phase_ratios_vertex[I...]

            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, _, λ_I, _, _, _ = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIvij)

            # update arrays
            τ_v[1][I...], τ_v[2][I...], τ_v[3][I...] = τxx_I, τyy_I, τxy_I
            ε_pl[3][I...] = εxy_pl
            λv[I...] = λ_I

            if do_partials
                Jτ_vertex = local_stress_jacobian_ε(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt, EIIvij)
                store_stress_jacobian_vertex!(dyrel, Jτ_vertex, I...)
            end

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
                EIIij = EII_pl[I...]
                ratio = phase_ratios_center[I...]

                # compute local stress
                τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I, ε_vol_pl_I =
                    compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EIIij)
                # update arrays
                τ[1][I...], τ[2][I...], τ[3][I...] = τxx_I, τyy_I, τxy_I
                ε_pl[1][I...], ε_pl[2][I...] = εxx_pl, εyy_pl
                ε_vol_pl[I...] = ε_vol_pl_I
                τII[I...] = τII_I
                η_vep[I...] = ηvep_I
                λ[I...] = λ_I
                ΔPψ[I...] = ΔPψ_I

                if do_partials
                    Jτ_center = local_stress_jacobian_ε(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt, EIIij)
                    store_stress_jacobian_center!(dyrel, Jτ_center, I...)
                    store_pressure_correction_jacobian!(∂ΔPψc_∂ε, Jτ_center, I...)
                end

            else
                τ[1][I...], τ[2][I...], τ[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_pl[1][I...], ε_pl[2][I...], ε_pl[3][I...] = 0.0e0, 0.0e0, 0.0e0
                ε_vol_pl[I...] = 0.0e0
                τII[I...] = 0.0e0
                η_vep[I...] = 0.0e0
                λ[I...] = 0.0e0
                ΔPψ[I...] = 0.0e0

            end
        end
    end

    return nothing
end
