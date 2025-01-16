@parallel_indices (I...) function update_stresses_center_vertex_psAD!(
    ε::NTuple{N, T},          # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{N, T},       # whole Voigt tensor @ centers
    EII,                      # accumulated plastic strain rate @ centers
    τ::NTuple{N, T},          # whole Voigt tensor @ centers
    τshear_v::NTuple{N2, T},  # shear tensor components @ vertices
    τ_o::NTuple{N, T},
    τshear_ov::NTuple{N2, T}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep, 
    relλ, 
    dt, 
    θ_dτ,
    rheology, 
    phase_center,
    phase_vertex,
) where {N, N2, T}


    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij       = av_clamped(Pr, Ic...)
    εxxv_ij     = av_clamped(ε[1], Ic...)
    εyyv_ij     = av_clamped(ε[2], Ic...)
    τxxv_ij     = av_clamped(τ[1], Ic...)
    τyyv_ij     = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij     = av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    if isinf(Kv)
        Kv = 0.0
    end
    volumev = Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ

    ηv_ij = av_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv = (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) * dτ_rv
    dτyyv = (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) * dτ_rv
    dτxyv = (-(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] + 2.0 * ηv_ij * ε[3][I...]) * dτ_rv
    #τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)


    # trial stress is known from forward solve
    τIIv_ij= av_clamped(τII, Ic...)
    
    # yield function @ center
    Fv = τIIv_ij - Cv - Pv_ij *sinϕv

    if is_pl && !iszero(τIIv_ij)

        if Fv .> 0.0
            λv[I...]    = (1.0 - relλ) * λv[I...] + relλ .* (Fv / (ηv_ij * dτ_rv + η_regv + volumev))
        elseif Fv .<= 0.0
            λv[I...]    = (1.0 - relλ) * λv[I...]
        end
        τIIv_ijAD = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
        dQdτxy      = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ijAD
        τxyv[I...] += dτxyv - 2.0 * ηv_ij  * 0.5 * λv[I...] * dQdτxy * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
    end


    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt  = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        K      = fn_ratio(get_bulk_modulus, rheology, phase)
        if isinf(K)
            K = 0.0
        end
        volume =  K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ

        ηij    = η[I...]
        dτ_r   = 1.0/(θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        #εij_ve = @. εij + 0.5*τij_o* _Gdt
        #εII_ve = GeoParams.second_invariant(εij_ve)
        # stress increments @ center
        dτij   = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        # trial stress is known from forward solve
        τII_ij= @inbounds τII[I...]



        # trial stress is known from forward solve
        #τII_ij= @inbounds τII[I...]
        # yield function @ center
        F  = τII_ij - C - Pr[I...] *sinϕ

        if is_pl && !iszero(τII_ij)

            # stress correction @ center
            #λ[I...] =
            #(1.0 - relλ) * λ[I...] +
            #relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            if F .> 0.0
                λ[I...]    = (1.0 - relλ) * λ[I...] + relλ .* (F / (η[I...] * dτ_r + η_reg + volume))
            else
                λ[I...]    = (1.0 - relλ) * λ[I...]
            end
            τII_ijAD = GeoParams.second_invariant(dτij .+ τij)
            dQdτij      = @. 0.5 * (τij + dτij) / τII_ijAD
            # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            εij_pl      =  λ[I...] .* dQdτij 
            dτij        = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij         = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            #τII[I...]   = GeoParams.second_invariant(τij)
            #Pr_c[I...]  = Pr[I...] + K * dt * λ[I...] * sinψ
            #η_vep[I...] = 0.5 * τII_ij / εII_ve
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            #η_vep[I...] = ηij
            #τII[I...]   = τII_ij
        end
    
        #Pr_c[I...]  = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end



@parallel_indices (I...) function assemble_parameter_matrices!(
    EII,
    Gv,
    Gc,
    ϕv,
    ϕc,
    Cv,
    Cc,
    rheology, 
    phase_center,
    phase_vertex,

)

    ni = size(Gc)

    Ic       = clamped_indices(ni, I...)
    EIIv_ij  = av_clamped(EII, Ic...)
    phase    = @inbounds phase_vertex[I...]
    Gv[I...] = fn_ratio(get_shear_modulus, rheology, phase)

    is_pl, Cvi, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    ϕv[I...] =  sinϕv #asind(sinϕv)
    Cv[I...] = Cvi

    if all(I .≤ ni)
        phase = @inbounds phase_center[I...]
        Gc[I...] = fn_ratio(get_shear_modulus, rheology, phase)

        is_pl, Cci, sinϕc, cosϕc, sinψc, η_regc = plastic_params_phase(rheology, EII[I...], phase)
        ϕc[I...] = sinϕc # asind(sinϕc)
        Cc[I...] = Cci
    end

return nothing

end

# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex_psSensTest!(
    ε::NTuple{3},         # normal components @ centers; shear components @ vertices
    ε_pl::NTuple{3},      # whole Voigt tensor @ centers
    EII,                  # accumulated plastic strain rate @ centers
    τ::NTuple{3},         # whole Voigt tensor @ centers
    τshear_v::NTuple{1},  # shear tensor components @ vertices
    τ_o::NTuple{3},
    τshear_ov::NTuple{1}, # shear tensor components @ vertices
    Pr,
    Pr_c,
    η,
    λ,
    λv,
    τII,
    η_vep,
    relλ,
    dt,
    θ_dτ,
    rheology,
    phase_center,
    phase_vertex,
    Gv,
    Gc,
    ϕv,
    ϕc,
    Cv,
    Cc,
)
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex
    Pv_ij       = av_clamped(Pr, Ic...)
    εxxv_ij     = av_clamped(ε[1], Ic...)
    εyyv_ij     = av_clamped(ε[2], Ic...)
    τxxv_ij     = av_clamped(τ[1], Ic...)
    τyyv_ij     = av_clamped(τ[2], Ic...)
    τxxv_old_ij = av_clamped(τ_o[1], Ic...)
    τyyv_old_ij = av_clamped(τ_o[2], Ic...)
    EIIv_ij     = av_clamped(EII, Ic...)

    ## vertex
    phase = @inbounds phase_vertex[I...]
    Gv_ij       = @inbounds Gv[I...]
    ϕv_ij       = @inbounds ϕv[I...]
    Cv_ij       = @inbounds Cv[I...]
    is_pl, Cv, sinϕvNot, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
    _Gvdt = inv(Gv_ij * dt)
    sinϕv = ϕv_ij
    #cosϕv = cosd(ϕv_ij)
    Cv    = Cv_ij

    Kv = fn_ratio(get_bulk_modulus, rheology, phase)
    if isinf(Kv)
        Kv = 0.0
    end
    
    volumev = Kv * dt * sinϕv * sinψv # plastic volumetric #change K * dt * sinϕ * sinψ

    ηv_ij = av_clamped(η, Ic...)
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    # stress increments @ vertex
    dτxxv =
        (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) * dτ_rv
    dτyyv =
        (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) * dτ_rv
    dτxyv =
        (
            -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
            2.0 * ηv_ij * ε[3][I...]
        ) * dτ_rv
    #τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)


    # trial stress is known from forward solve
    τIIv_ij= av_clamped(τII, Ic...)

    # yield function @ center
    Fv = τIIv_ij - Cv - Pv_ij * sinϕv
    if is_pl && !iszero(τIIv_ij)
        # stress correction @ vertex
        if Fv .> 0.0
            λv[I...]    = (1.0 - relλ) * λv[I...] + relλ .* (Fv / (ηv_ij * dτ_rv + η_regv + volumev))
        else
            λv[I...]    = (1.0 - relλ) * λv[I...]
        end
        τIIv_ijAD = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ijAD
        #λv[I...] = (1.0 - relλ) * λv[I...] + relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        #dQdτxy = 1.0
        τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
    end

    ## center
    if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        Gc_ij       = @inbounds Gc[I...]
        ϕc_ij       = @inbounds ϕc[I...]
        Cc_ij       = @inbounds Cc[I...]
        _Gcdt = inv(Gc_ij * dt)
        is_pl, C, sinϕNot, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        sinϕ = ϕc_ij
        #cosϕ = cosd(ϕv_ij)
        C    = Cc_ij

        K = fn_ratio(get_bulk_modulus, rheology, phase)
        if isinf(K)
            K = 0.0
        end
        volume = K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ + ηij * _Gcdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gcdt
        εII_ve = GeoParams.second_invariant(εij_ve)
        # stress increments @ center
        dτij = @. (-(τij - τij_o) * ηij * _Gcdt - τij .+ 2.0 * ηij * εij) * dτ_r

        # trial stress is known from forward solve
        τII_ij= @inbounds τII[I...]

        # yield function @ center
        F = τII_ij - C - Pr[I...] * sinϕ

        if is_pl && !iszero(τII_ij)
            # stress correction @ center
            if F .> 0.0
                λ[I...]    = (1.0 - relλ) * λ[I...] + relλ .* (F / (η[I...] * dτ_r + η_reg + volume))
            else
                λ[I...]    = (1.0 - relλ) * λ[I...]
            end
            τII_ijAD = GeoParams.second_invariant(dτij .+ τij)
            dQdτij = @. 0.5 * (τij + dτij) / τII_ijAD
            # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            #τII[I...] = GeoParams.second_invariant(τij)
            #Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
            #η_vep[I...] = 0.5 * τII_ij / εII_ve
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            #η_vep[I...] = ηij
            #τII[I...] = τII_ij
        end

        #Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end