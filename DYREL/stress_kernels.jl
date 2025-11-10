function compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation, dt)
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) compute_stress_DRYEL!(stokes, rheology, phase_ratios.center, phase_ratios.vertex, λ_relaxation, dt)
    return nothing
end

@parallel_indices (I...) function compute_stress_DRYEL!(stokes::JustRelax.StokesArrays, rheology, phase_ratios_center, phase_ratios_vertex, λ_relaxation, dt)

    Base.@propagate_inbounds @inline av(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ni = size(phase_ratios_center)

    ## VERTEX CALCULATION
    # @inbounds begin
        Ic    = clamped_indices(ni, I...)
        # τij_o = av_clamped(stokes.τ_o.xx, Ic...), av_clamped(stokes.τ_o.yy, Ic...), stokes.τ_o.xy[I...]
        τij_o = stokes.τ_o.xx_v[I...], stokes.τ_o.yy_v[I...], stokes.τ_o.xy[I...]
        εij   = av_clamped(stokes.ε.xx, Ic...), av_clamped(stokes.ε.yy, Ic...), stokes.ε.xy[I...]
        λvij  = stokes.λv[I...]
        ηij   = JustRelax2D.harm_clamped(stokes.viscosity.η, Ic...)
        Pij   = av_clamped(stokes.P, Ic...)
        ratio = phase_ratios_vertex[I...]
        # # compute local stress
        τxx_I, τyy_I, τxy_I, _, _, _, _, λ_I, = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt)

        # update arrays
        stokes.τ.xx_v[I...], stokes.τ.yy_v[I...], stokes.τ.xy[I...] = τxx_I, τyy_I, τxy_I
        stokes.λv[I...]   = λ_I

        ## CENTER CALCULATION
        if all(I .≤ ni)
            τij_o = stokes.τ_o.xx[I...], stokes.τ_o.yy[I...], stokes.τ_o.xy_c[I...]
            εij   = stokes.ε.xx[I...], stokes.ε.yy[I...], av(stokes.ε.xy)
            λij   = stokes.λ[I...]
            ηij   = stokes.viscosity.η[I...]
            Pij   = stokes.P[I...]
            ratio = phase_ratios_center[I...]
            
            # compute local stress
            τxx_I, τyy_I, τxy_I, εxx_pl, εyy_pl, εxy_pl, τII_I, λ_I, ΔPψ_I, ηvep_I = compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt)
            # update arrays
            stokes.τ.xx[I...],    stokes.τ.yy[I...],   stokes.τ.xy_c[I...]     = τxx_I, τyy_I, τxy_I
            stokes.ε_pl.xx[I...], stokes.ε_pl.yy[I...], stokes.ε_pl.xy_c[I...] = εxx_pl, εyy_pl, εxy_pl
            stokes.τ.II[I...]            = τII_I
            stokes.viscosity.η_vep[I...] = ηvep_I
            stokes.λ[I...]               = λ_I
            stokes.ΔPψ[I...]             = ΔPψ_I
        end
    # end

    return nothing
end

@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt) where {N}
    quote
        @inline
        # iterate over phases
        v_phases = Base.@ntuple $N phase -> begin
            # get phase ratio
            ratio_I = phase_ratio[phase]
            v = if iszero(ratio_I) # this phase does not contribute
                empty_stress_solution(εij)

            else

                # get rheological properties for this phase
                G  = get_shear_modulus(rheology, phase)
                Kb = get_bulk_modulus(rheology, phase)
                ispl, C, sinϕ, cosϕ, sinΨ, η_reg = JustRelax2D.plastic_params(rheology[phase].CompositeRheology[1].elements, 0e0)
                # compute local stress
                _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, ratio_I, dt)
            end
        end
        # sum contributions from all phases
        v = reduce(.+, v_phases)
        return v # this returns (τ_ij...), (εij_pl...), τII, λ, ΔPψ, ηvep
    end
end

function _compute_local_stress(εij, τij_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, phase_ratio, dt)

    # viscoelastic viscosity
    η_ve  = inv(inv(η) + inv(G * dt))
    # effectice stgrain rate
    _Gdt2 = inv(2 * G * dt)
    εij_eff = @. εij + τij_o * _Gdt2

    εII  = second_invariant(εij_eff)
    # early return if there is no deformation
    iszero(εII) && return (zero_tuple(εij)..., zero_tuple(εij)..., 0.0, 0.0, 0.0, η)

    # Plastic stress correction starts here
    τij   = @. 2 * η_ve * εij_eff
    τII   = second_invariant(τij)
    # Plasticity
    # F     = τII - C * cosϕ - max(P, 0) * sinϕ
    F     = τII - C * cosϕ - P * sinϕ
    λ     = if ispl && F > 0
        λ_new = F / (η_ve + η_reg + Kb * dt * sinϕ * sinΨ)
        λ_relaxation * λ_new + (1 - λ_relaxation) * λ

    else
        
        0.0
    end

    # Effective viscoelastic-plastic viscosity
    η_vep  = (τII - λ * η_ve) / (2 * εII) * phase_ratio
    # Update stress and plastic strain rate
    τij    = @. 2 * η_vep * εij_eff * phase_ratio
    τII    = second_invariant(τij)
    dQdτij = @. 0.5 * τij / τII
    εij_pl = @. λ * dQdτij * phase_ratio
    # Update pressure correction due to dilatation
    ΔPψ    = iszero(sinΨ) ? 0.0 : λ * sinΨ * Kb * dt * phase_ratio

    return τij..., εij_pl..., τII, λ * phase_ratio, ΔPψ, η_vep
end

# this returns zero for: τxx, τyy, τxy, τII, ηvep, λ, ΔPψ
@inline empty_stress_solution(::NTuple{3, T}) where T = zero_tuple(T, Val(10))
# this returns zero for: τxx, τyy, τyy, τyz, τxz, τxy, τII, ηvep, λ, ΔPψ
@inline empty_stress_solution(::NTuple{6, T}) where T = zero_tuple(T, Val(13)) 

@inline zero_tuple(::Type{T}, ::Val{N}) where {T, N} = ntuple(_-> zero(T), Val(N))
@inline zero_tuple(::NTuple{N, T}) where {T, N} = zero_tuple(T, Val(N))


