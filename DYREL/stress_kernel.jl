@generated function compute_local_stress(εij, τij_o, η, P, λ, λ_relaxation, rheology, phase_ratio::SVector{N}, dt) where N
    quote
        @inline
        v_phases = Base.@ntuple $N phase -> begin
            ratio_I = phase_ratio[phase]
            v =  if iszero(ratio_I)  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            else
                G  = fn_ratio(get_shear_modulus, rheology, phase)
                Kb = fn_ratio(get_bulk_modulus, rheology, phase)
                ispl, C, sinϕ, cosϕ, sinΨ, η_reg = JustRelax2D.plastic_params(rheology[phase].CompositeRheology[1].elements, 0e0)
            
                _compute_local_stress(εij..., τij_o..., η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, ratio_I, dt)
            end
        end
        # this returns τxx, τyy, τxy, τII, ηvep, λ, ΔPψ
        return reduce(.+, v_phases)
    end
end

function _compute_local_stress(εxx, εyy, εxy, τxx_o, τyy_o, τxy_o, η, P, G, Kb, λ, λ_relaxation, ispl, C, sinϕ, cosϕ, sinΨ, η_reg, phase_ratio, dt)

    εII  = second_invariant(εxx, εyy, εxy)
    # early return if no deformation
    iszero(εII) && return (0.0, 0.0, 0.0, 0.0, η, 0.0, 0.0)

    # viscoelastic viscosity
    η_ve  = inv(inv(η) + inv(G * dt))
    # Deviatoric stress
    _Gdt2 = inv(2 * G * dt)
    τxx   = 2 * η_ve * (εxx + τxx_o * _Gdt2)
    τyy   = 2 * η_ve * (εyy + τyy_o * _Gdt2)
    τxy   = 2 * η_ve * (εxy + τxy_o * _Gdt2)
    τij   = τxx, τyy, τxy
    τII   = second_invariant(τij)
    # Plasticity
    F     = τII - C * cosϕ - max(P, 0) * sinϕ
    λ     = if ispl && F > 0
        λ_new = max(F, 0.0) / (η_ve + η_reg + Kb * dt * sinϕ * sinΨ)
        (1.0 - λ_relaxation) * λ + λ_relaxation * λ_new
    else
        λ     = 0.0
    end

    ηvep  = (τII - λ * η_ve) / (2 * εII) * phase_ratio
    τxx   = 2 * ηvep * (εxx + τxx_o * _Gdt2) * phase_ratio
    τyy   = 2 * ηvep * (εyy + τyy_o * _Gdt2) * phase_ratio
    τxy   = 2 * ηvep * (εxy + τxy_o * _Gdt2) * phase_ratio

    ΔPψ   = iszero(sinΨ) ? 0.0 : λ * sinΨ * Kb * dt * phase_ratio

    τij   = τxx, τyy, τxy
    τII   = second_invariant(τij)

    return τxx, τyy, τxy, τII, ηvep, λ * phase_ratio, ΔPψ
end

function compute_DRYEL_stress(stokes::JustRelax.StokesArrays, rheology, phase_ratios, dt)

    Base.@propagate_inbounds @inline av_shear(A) = sum(JustRelax2D._gather(A, I...)) / 4

    ## CENTER CALCULATION
    τij   = stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.xy_c[I...]
    τij_o = stokes.τ_o.xx[I...], stokes.τ_o.yy[I...], stokes.τ_o.xy[I...]
    εij   = stokes.ε.xx[I...], stokes.ε.yy[I...],  av_shear(stokes.ε.xy)
    λij   = stokes.λ[I...]
    ηij   = stokes.viscosity.η[I...]
    Pij   = stokes.P[I...]
    ratio = phase_ratios.center[I...]

    # compute local stress
    τxx_I, τyy_I, τxy_I, τII_I, ηvep_I, λ_I, ΔPψ_I = compute_local_stress(εij, τij_o, ηij, Pij, λij, λ_relaxation, rheology, ratio, dt)
    # update arrays
    stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.xy_c[I...] = τxx_I, τyy_I, τxy_I
    stokes.τ.II[I...]            = τII_I
    stokes.viscosity.η_vep[I...] = ηvep_I
    stokes.λ[I...]               = λ_I
    stokes.ΔPψ[I...]             = ΔPψ_I

    ## VERTEX CALCULATION (TO BE FINISHED)
    τij    = stokes.τ.xx[I...], stokes.τ.yy[I...], stokes.τ.xy_c[I...]
    τij_o  = stokes.τ_o.xx[I...], stokes.τ_o.yy[I...], stokes.τ_o.xy[I...]
    εij    = stokes.ε.xx[I...], stokes.ε.yy[I...],  av_shear(stokes.ε.xy)
    λvij   = stokes.λv[I...]
    ηij    = stokes.viscosity.η[I...]
    Pij    = stokes.P[I...]

    ratio = phase_ratios.vertex[I...]
    # compute local stress
    _, _, τxy_I, _, _, λ_I, _ = compute_local_stress(εij, τij_o, ηij, Pij, λvij, λ_relaxation, rheology, ratio, dt)
    # update arrays
    stokes.τ.xy[I...] = τxy_I
    stokes.λv[I...]   = λ_I
    
    return nothin
end

# lets do center first
I = i, j = 1, 1

λ_relaxation = 1
