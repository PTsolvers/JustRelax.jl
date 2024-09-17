
function update_stresses_center_vertex!(
    ε::NTuple{N, T},          # normal components @ centers; shear components @ vertices
    τ::NTuple{N, T},          # whole Voigt tensor @ centers
    τshear_v::NTuple{N, T},   # shear tensor components @ vertices
    τ_o::NTuple{N, T},
    τshear_ov::NTuple{N, T}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    τII,
    η_vep, 
    Fchk, 
    relλ, dt, re_mech, vdτ, lτ, r,
    rheology, 
    phase_center,
    phase_vertex,
) where {N, T}

    θ_dτ  = lτ*(r+2.0)/(re_mech*vdτ)

    τxyv     = τshear_v[1]
    τxyv_old = τshear_ov[1]

    for j in axes(Pr,2), i in axes(Pr,1)
        I   = i,j
        Ic  = clamped_indices(ni, i, j)
        
        # Material properties
        ## center
        phase = @inbounds phase_center[I...]
        _Gdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = JR.plastic_params_phase(rheology, 0e0, phase)
        K      = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij    = η[I...]
        dτ_r   = 1.0/(θ_dτ + ηij * _Gdt + 1.0)

        ## vertex
        phase  = @inbounds phase_vertex[I...]
        _Gvdt  = inv(fn_ratio(JR.get_shear_modulus, rheology, phase) * dt)
        _, Cv, sinϕv, cosϕv, sinψv, η_regv = JR.plastic_params_phase(rheology, 0e0, phase)
        Kv     = fn_ratio(JR.get_bulk_modulus, rheology, phase)
        volumev= isinf(K) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij  = av_clamped_indices(η, Ic...)
        dτ_rv  = 1.0/(θ_dτ + ηv_ij * _Gdt + 1.0)

        # interpolate to ith vertex
        Pv_ij       = av_clamped_indices(Pr, Ic...)
        εxxv_ij     = av_clamped_indices(εxx, Ic...)
        εyyv_ij     = av_clamped_indices(εyy, Ic...)
        τxxv_ij     = av_clamped_indices(τxx, Ic...)
        τyyv_ij     = av_clamped_indices(τyy, Ic...)
        τxxv_old_ij = av_clamped_indices(τxx_old, Ic...)
        τyyv_old_ij = av_clamped_indices(τyy_old, Ic...)

        # cache strain rates for center calculations
        τij, τij_o, εij = JR.cache_tensors(τ, τ_o, ε, i, j)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5*τij_o* _Gdt
        εII_ve = GP.second_invariant(εij_ve)
        # stress increments @ center
        dτij   = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        τII_ij = GP.second_invariant(dτij .+ τij)

        # stress increments @ vertex
        dτxxv   = (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) * dτ_rv
        dτyyv   = (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) * dτ_rv
        dτxyv   = (-(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] + 2.0 * ηv_ij * εxyv[I...]) * dτ_rv
        τIIv_ij = √(0.5*((τxxv_ij  + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
         
        # yield function @ center
        F  = τII_ij - C - Pr[I...] *sinϕ
        # yield function @ center
        Fv = τIIv_ij - Cv - Pv_ij *sinϕv
        if is_pl #&& F > 0
            # stress correction @ center
            λ[I...]     = (1.0 - relλ)*λ[I...]  + relλ.*(max(F,0.0)/(η[I...] *dτ_r + η_reg + volume))
            dQdτij      = @. 0.5 * (τij + dτij) / τII_ij
            dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            τij         = dτij .+ τij
            setindex!.(τ, τij, I...)
            τII[I...]   = GP.second_invariant(τij)
            Pr_c[I...]  = Pr[I...] + K*dt*λ[I...] * sinψ
            Fchk[I...]  = τII_ij - τ_y - Pr_c[I...]*sinϕ - λ[I...]*η_reg
            η_vep[I...] = τII_ij / 2.0 / εII_ve

            # stress correction @ vertex
            λv[I...]    = (1.0 - relλ) * λv[I...] + relλ*(max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy      = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] += (-(τxyv[I...] - τxyv_old[I...] ) * ηv_ij * _Gvdt - τxyv[I...]  + 2.0 * ηv_ij  *(εxyv[I...] - 0.5 * λv[I...] * dQdτxy)) * dτ_rv
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            Fchk[I...]  = 0e0
            η_vep[I...] = ηij
            τII[I...]   = τII_ij

            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
        
        Pr_c[I...]  = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end
    
    return
end