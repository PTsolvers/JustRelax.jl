@parallel_indices (I...) function update_stresses_center_vertexAD!(
        ε::NTuple{3, T},      # normal components @ centers; shear components @ vertices
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
        ϕ::JustRelax.RockRatio,
    ) where {T}
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij = av_clamped(Pr, Ic...)
        εxxv_ij = av_clamped(ε[1], Ic...)
        εyyv_ij = av_clamped(ε[2], Ic...)
        τxxv_ij = av_clamped(τ[1], Ic...)
        τyyv_ij = av_clamped(τ[2], Ic...)
        τxxv_old_ij = av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped(τ_o[2], Ic...)
        EIIv_ij = av_clamped(EII, Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv =
            (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) *
            dτ_rv
        dτyyv =
            (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) *
            dτ_rv
        dτxyv =
            (
            -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
                2.0 * ηv_ij * ε[3][I...]
        ) * dτ_rv

        τIIv_ij = √(
            0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2
        )

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(
                rheology, EII[I...], phase
            )
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
            εij_ve = @. εij + 0.5 * τij_o * _Gdt
            εII_ve = GeoParams.second_invariant(εij_ve)
            # stress increments @ center
            dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r

            τII_ij = GeoParams.second_invariant(dτij .+ τij)
           # yield function @ center
            F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
                setindex!.(ε_pl, εij_pl, I...)
                τII[I...] = τII_ij = GeoParams.second_invariant(τij)
                # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                # η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)
                # η_vep[I...] = ηij
#                τII[I...] = τII_ij
            end
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
#            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
        else
            Pr_c[I...] = zero(eltype(T))
            # τij, = cache_tensors(τ, τ_o, ε, I...)
            dτij = zero(eltype(T)), zero(eltype(T)), zero(eltype(T))
            setindex!.(τ, dτij .+ τij, I...)
        end
    end

    return nothing
end

@parallel_indices (I...) function update_stresses_center_vertexADSens!(
        ε::NTuple{3, T},      # normal components @ centers; shear components @ vertices
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
        ϕ::JustRelax.RockRatio,
        Sens
    ) where {T}
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij = av_clamped(Pr, Ic...)
        εxxv_ij = av_clamped(ε[1], Ic...)
        εyyv_ij = av_clamped(ε[2], Ic...)
        τxxv_ij = av_clamped(τ[1], Ic...)
        τyyv_ij = av_clamped(τ[2], Ic...)
        τxxv_old_ij = av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped(τ_o[2], Ic...)
        EIIv_ij = av_clamped(EII, Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, CvNot, sinϕvNot, cosϕvNot, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        #_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)

        Gv_ij = av_clamped(Sens[1],Ic...)
        ϕv_ij = av_clamped(Sens[2],Ic...)
        Cv_ij = av_clamped(Sens[3],Ic...)
        _Gvdt = inv(Gv_ij * dt)
        sinϕv = sind(ϕv_ij)
        cosϕv = cosd(ϕv_ij)
        Cv    = Cv_ij

        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
 #       θ_dτ = 1.0
 #       dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
 #       dτxxv =
 #           (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij #* εxxv_ij)# *
 #           dτ_rv
 #       dτyyv =
 #           (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij #* εyyv_ij)# *
 #           dτ_rv
################################
#### NEW ####
 
#=        dτxyv =
            (
            -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
                2.0 * ηv_ij * ε[3][I...]
        ) * dτ_rv
=#
        dτxyv = ((ε[3][I...] + (τxyv_old[I...]*0.5*_Gvdt))*inv((0.5*_Gvdt + 0.5*inv(ηv_ij))))
################################

#        τIIv_ij = √(
#            0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2
#        )
#=
        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
        else
=#            # stress correction @ vertex
            τxyv[I...] = dτxyv
        #end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            #_Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, CNot, sinϕNot, cosϕNot, sinψ, η_reg = plastic_params_phase(
                rheology, EII[I...], phase
            )
            Gc_ij       = Sens[1][I...]
            ϕc_ij       = Sens[2][I...]
            Cc_ij       = Sens[3][I...]
            _Gdt = inv(Gc_ij * dt)
            sinϕ = sind(ϕc_ij)
            cosϕ = cosd(ϕc_ij)
            C    = Cc_ij

            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
#            θ_dτ = 1.0
#            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = GeoParams.second_invariant(εij_ve)
################################
#### NEW ####

            # stress increments @ center
 #           dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r

            # stress increments @ center
            dτij = @. ((εij + (τij_o * 0.5 * _Gdt))*(inv(0.5*_Gdt + 0.5*inv(ηij))))
################################


#            τII_ij = GeoParams.second_invariant(dτij .+ τij)
#=           # yield function @ center
            F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
                setindex!.(ε_pl, εij_pl, I...)
                τII[I...] = τII_ij = GeoParams.second_invariant(τij)
                # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                # η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
=#                # stress correction @ center
            setindex!.(τ, dτij, I...) # setindex!.(τ, dτij .+ τij, I...)
                # η_vep[I...] = ηij
#                τII[I...] = τII_ij
            #end
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
#            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
        else
            Pr_c[I...] = zero(eltype(T))
            # τij, = cache_tensors(τ, τ_o, ε, I...)
            dτij = zero(eltype(T)), zero(eltype(T)), zero(eltype(T))
            # setindex!.(τ, dτij .+ τij, I...)
            setindex!.(τ, dτij, I...)
        end
    end

    return nothing
end

#=
@parallel_indices (I...) function update_stresses_center_vertexAD!(
        ε::NTuple{3, T},      # normal components @ centers; shear components @ vertices
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
        ϕ::JustRelax.RockRatio,
    ) where {T}
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij = av_clamped(Pr, Ic...)
        εxxv_ij = av_clamped(ε[1], Ic...)
        εyyv_ij = av_clamped(ε[2], Ic...)
        τxxv_ij = av_clamped(τ[1], Ic...)
        τyyv_ij = av_clamped(τ[2], Ic...)
        τxxv_old_ij = av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped(τ_o[2], Ic...)
        EIIv_ij = av_clamped(EII, Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv =
            (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij) *
            dτ_rv
        dτyyv =
            (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij) *
            dτ_rv
        dτxyv =
            (
            -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
                2.0 * ηv_ij * ε[3][I...]
        ) * dτ_rv
#        τIIv_ij = √(
#            0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + #dτxyv)^2
#        )
        τIIv_ij = av_clamped(τII, Ic...)
        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(
                rheology, EII[I...], phase
            )
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = GeoParams.second_invariant(εij_ve)
            # stress increments @ center
            dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
#            τII_ij = GeoParams.second_invariant(dτij .+ τij)
            τII_ij = τII[I...]
           # yield function @ center
            F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
                setindex!.(ε_pl, εij_pl, I...)
#                τII[I...] = τII_ij = GeoParams.second_invariant(τij)
                # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                # η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)
                # η_vep[I...] = ηij
                τII[I...] = τII_ij
            end
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
        else
            Pr_c[I...] = zero(eltype(T))
            # τij, = cache_tensors(τ, τ_o, ε, I...)
            dτij = zero(eltype(T)), zero(eltype(T)), zero(eltype(T))
            # setindex!.(τ, dτij .+ τij, I...)
            setindex!.(τ, dτij, I...)
        end
    end

    return nothing
end
=#

#=
@parallel_indices (I...) function update_stresses_center_vertexADSens!(
        ε::NTuple{3, T},      # normal components @ centers; shear components @ vertices
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
        ϕ::JustRelax.RockRatio,
        Sens
    ) where {T}
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij = av_clamped(Pr, Ic...)
        εxxv_ij = av_clamped(ε[1], Ic...)
        εyyv_ij = av_clamped(ε[2], Ic...)
        τxxv_ij = av_clamped(τ[1], Ic...)
        τyyv_ij = av_clamped(τ[2], Ic...)
        τxxv_old_ij = av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped(τ_o[2], Ic...)
        EIIv_ij = av_clamped(EII, Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, CvNot, sinϕvNot, cosϕvNot, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        #_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)

        Gv_ij = av_clamped(Sens[1],Ic...)
        ϕv_ij = av_clamped(Sens[2],Ic...)
        Cv_ij = av_clamped(Sens[3],Ic...)
        _Gvdt = inv(Gv_ij * dt)
        sinϕv = sind(ϕv_ij)
        cosϕv = cosd(ϕv_ij)
        Cv    = Cv_ij

        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
#        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
#        dτxxv =
#            (-(τxxv_ij - τxxv_old_ij) * ηv_ij * _Gvdt - τxxv_ij + 2.0 * ηv_ij * εxxv_ij)# * dτ_rv
#        dτyyv =
#            (-(τyyv_ij - τyyv_old_ij) * ηv_ij * _Gvdt - τyyv_ij + 2.0 * ηv_ij * εyyv_ij)# * dτ_rv

####################################
#=
        dτxyv =
            (
            -(τxyv[I...] - τxyv_old[I...]) * ηv_ij * _Gvdt - τxyv[I...] +
                2.0 * ηv_ij * ε[3][I...]
        )# * dτ_rv
=#
        #### NEW ####
        dτxyv = ((ε[3][I...] + τxyv_old[I...] * 0.5 * _Gvdt) * (inv(_Gvdt + inv(ηv_ij))))
####################################        

#        τIIv_ij = √(
#            0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2
#        )
#=        τIIv_ij = av_clamped(τII, Ic...)

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            τxyv[I...] = dτxyv - 2.0 * ηv_ij * λv[I...] * dQdτxy * dτ_rv
        else
=#            # stress correction @ vertex
            τxyv[I...] = dτxyv
#        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, CNot, sinϕNot, cosϕNot, sinψ, η_reg = plastic_params_phase(
                rheology, EII[I...], phase
            )
            Gc_ij       = Sens[1][I...]
            ϕc_ij       = Sens[2][I...]
            Cc_ij       = Sens[3][I...]
            _Gdt = inv(Gc_ij * dt)
            sinϕ = sind(ϕc_ij)
            cosϕ = cosd(ϕc_ij)
            C    = Cc_ij

            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
#            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = GeoParams.second_invariant(εij_ve)

################################
#### NEW ####
#=
            # stress increments @ center
            dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij)# * dτ_r
=#
            # stress increments @ center
            dτij = @. ((εij + τij_o * 0.5 * _Gdt) * (inv(_Gdt + inv(ηij))))
################################

#            τII_ij = GeoParams.second_invariant(dτij .+ τij)
#=            τII_ij = τII[I...]
           # yield function @ center
            F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ .* (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
                setindex!.(ε_pl, εij_pl, I...)
#                τII[I...] = τII_ij = GeoParams.second_invariant(τij)
                # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                # η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
=#                # stress correction @ center
                #setindex!.(τ, dτij .+ τij, I...)
                setindex!.(τ, dτij, I...)
                # η_vep[I...] = ηij
#                τII[I...] = τII_ij
#            end
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
#            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
        else
            Pr_c[I...] = zero(eltype(T))
            # τij, = cache_tensors(τ, τ_o, ε, I...)
            dτij = zero(eltype(T)), zero(eltype(T)), zero(eltype(T))
            # setindex!.(τ, dτij .+ τij, I...)
            setindex!.(τ, dτij, I...)
        end
    end

    return nothing
end
=#