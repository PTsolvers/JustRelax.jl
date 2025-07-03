@parallel_indices (I...) function dτdV_viscoelastic(
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
        iter,
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
  is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
  _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
  Kv = fn_ratio(get_bulk_modulus, rheology, phase)
  volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
  ηv_ij = av_clamped(η, Ic...)
  dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

    #### Analytical ####
    # (2.0ηij) / (1.0 + θ_dτ + _Gdt*ηij)
    dtv     = inv(1.0 + θ_dτ + _Gvdt*ηv_ij)
    dτdεxyv = (2.0*ηv_ij) * dtv        # ((iter+1)/2)*
    dtvn     = inv(θ_dτ + _Gvdt*ηv_ij)
    #dτdτv   = -(_Gvdt*ηv_ij) * dtvn    #(-1.0 - _Gvdt*ηv_ij)*dtv # (-1 - _Gdt*ηij) / (1.0 + θ_dτ + _Gdt*ηij)
    dτdτv = (-1.0 - _Gvdt*ηv_ij)*dtv
    ε[3][I...] = ((τxyv[I...])*(dτdεxyv))
    τxyv[I...] = τxyv[I...] + (τxyv[I...]*dτdτv)
    ####################
  end

    ## center
    if all(I .≤ ni)
          # Material properties
          phase = @inbounds phase_center[I...]
          _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
          ηij = η[I...]
          dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        #### Analytical ####
        # (2.0ηij) / (1.0 + θ_dτ + _Gdt*ηij)
        dtc   = inv(1.0 + θ_dτ + _Gdt*ηij)
        dτdε = (2.0*ηij) * dtc          # ((iter+1)/2)*
        dtcn = inv(θ_dτ + _Gdt*ηij)
        #dτdτ = -(_Gdt*ηij) * dtcn #(-1.0 - _Gdt*ηij) * dtc  # (-1 - _Gdt*ηij) / (1.0 + θ_dτ + _Gdt*ηij)
        dτdτ = (-1.0 - _Gdt*ηij) * dtc

        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)
        εxx = (τij[1])*(dτdε)
        εyy = (τij[2])*(dτdε)
        ε[1][I...] = εxx
        ε[2][I...] = εyy
        setindex!.(τ, τij .+ (τij.*dτdτ), I...)

        ####################
    end

    return nothing
end

@parallel_indices (I...) function dτdη_viscoelastic(
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
        τAD::NTuple{3},         # whole Voigt tensor @ centers
        τshear_vAD::NTuple{1},  # shear tensor components @ vertices
    ) where {T}
    τxyv   = τshear_v[1]
    τxyvAD = τshear_vAD[1]
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
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)

        dτdηv = (2.0*(ε[3][I...] + 0.5*_Gvdt*τxyv_old[I...])) / ((1.0 + _Gvdt*ηv_ij)^2.0)
        τxyv[I...] = τxyv[I...]*dτdηv

    ## center
    if all(I .≤ ni)
            # Material properties
            phase = @inbounds phase_center[I...]

            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            #Gdt = fn_ratio(get_shear_modulus, rheology, phase) * dt
            _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)

            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            dτdη = @zeros(3)
            
            dτdη = @. (2.0*(εij + 0.5*_Gdt*τij_o)) / ((1.0 + _Gdt*ηij)^2.0)
    #            τijAD, τij_o, εij = cache_tensors(τAD, τ_o, ε, I...)

        setindex!.(τ, τij .* dτdη, I...)
    end
end

    return nothing
end

# 2D kernel
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

    @inbounds if isvalid_v(ϕ, I...)
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
        phase = phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(
            τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
        )
        τijv = τxxv_ij, τyyv_ij, τxyv[I...]
        dτijv = dτxxv, dτyyv, dτxyv

        τIIv_ij = second_invariant(dτijv .+ τijv)
        #τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
        #τIIv_ij = av_clamped(τII, Ic...)

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv

        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            #λv[I...] =
            #    @muladd (1.0 - relλ) * λv[I...] +
            #    relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            #λvtemp = (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            Dv = (ηv_ij * dτ_rv + η_regv + volumev)
            λvtemp = (τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv) / (1+(ηv_ij/Dv))

            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            #εij_pl = λv[I...] * dQdτxy
            εij_pl = λvtemp * dQdτxy
            τxyv[I...] += @muladd dτxyv - 2.0 * ηv_ij * εij_pl * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        @inbounds if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = second_invariant(εij_ve)
            # stress increments @ center
            dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)

            τII_ij = second_invariant(dτij .+ τij)
            #τII_ij = √(0.5 * ((τij[1] + dτij[1])^2 + (τij[2] + dτij[2])^2) + (τij[3] + dτij[3])^2)
            #τII_ij = @inbounds τII[I...]
            # yield function @ center

            F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

            #τII_ij = 
            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                #λ[I...] =
                #    @muladd (1.0 - relλ) * λ[I...] +
                #    relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))

                #λtemp = (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))

                D = (η[I...] * dτ_r + η_reg + volume)
                λtemp = (τII_ij - C * cosϕ - Pr[I...] * sinϕ) / (1+(η[I...]/D))

                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                #εij_pl = λ[I...] .* dQdτij
                εij_pl = λtemp .* dQdτij

                dτij = @muladd @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                Base.@nexprs 3 i -> begin
                    τ[i][I...] = τij[i]
                    #ε_pl[i][I...] = εij_pl[i]
                end
                #τII_ij = second_invariant(τij)
            else
                # stress correction @ center
                Base.@nexprs 3 i -> begin
                    τ[i][I...] = dτij[i] + τij[i]
                end
                #τII_ij
            end
#            τII[I...] = τII_ij
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

        else
            Pr_c[I...] = zero(eltype(T))
            Base.@nexprs 3 i -> begin
                τ[i][I...] = zero(eltype(T))
            end

        end
    end

    return nothing
end

# 2D kernel
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
        Sens,
    ) where {T}
    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    @inbounds if isvalid_v(ϕ, I...)
        # interpolate to ith vertex
        Pv_ij = av_clamped(Pr, Ic...)
        εxxv_ij = av_clamped(ε[1], Ic...)
        εyyv_ij = av_clamped(ε[2], Ic...)
        τxxv_ij = av_clamped(τ[1], Ic...)
        τyyv_ij = av_clamped(τ[2], Ic...)
        τxxv_old_ij = av_clamped(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped(τ_o[2], Ic...)
        EIIv_ij = av_clamped(EII, Ic...)
        Gv = Sens[5][I...]
        Kv = Sens[8][I...]

        ## vertex
        phase = phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
        #_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        _Gvdt = inv(Gv * dt)
        #Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(
            τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
        )
        τijv = τxxv_ij, τyyv_ij, τxyv[I...]
        dτijv = dτxxv, dτyyv, dτxyv
        τIIv_ij = second_invariant(dτijv .+ τijv)
        #τIIv_ij = av_clamped(τII, Ic...)

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv

        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            #λv[I...] =
            #    @muladd (1.0 - relλ) * λv[I...] +
            #    relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            #λvtemp = (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            Dv = (ηv_ij * dτ_rv + η_regv + volumev)
            λvtemp = (τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv) / (1+(ηv_ij/Dv))

            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij
            #εij_pl = λv[I...] * dQdτxy
            εij_pl = λvtemp * dQdτxy
            τxyv[I...] += @muladd dτxyv - 2.0 * ηv_ij * εij_pl * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    else
        τxyv[I...] = zero(eltype(T))
    end

    ## center
    if all(I .≤ ni)
        @inbounds if isvalid_c(ϕ, I...)
            # Material properties
            phase = @inbounds phase_center[I...]
            G     = @inbounds Sens[1][I...]
            K     = @inbounds Sens[4][I...]
            _Gdt = inv(G * dt)
            #_Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
            #K = fn_ratio(get_bulk_modulus, rheology, phase)
            volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = second_invariant(εij_ve)
            # stress increments @ center
            dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
            τII_ij = second_invariant(dτij .+ τij)
            #τII_ij = τII[I...]
            # yield function @ center
            F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

            #τII_ij = 
            if is_pl && !iszero(τII_ij) && F > 0
                # stress correction @ center
                #λ[I...] =
                #    @muladd (1.0 - relλ) * λ[I...] +
                #    relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))

                #λtemp = (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))

                D = (η[I...] * dτ_r + η_reg + volume)
                λtemp = (τII_ij - C * cosϕ - Pr[I...] * sinϕ) / (1+(η[I...]/D))

                dQdτij = @. 0.5 * (τij + dτij) / τII_ij
                #εij_pl = λ[I...] .* dQdτij
                εij_pl = λtemp .* dQdτij

                dτij = @muladd @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                Base.@nexprs 3 i -> begin
                    τ[i][I...] = τij[i]
                    #ε_pl[i][I...] = εij_pl[i]
                end
                #τII_ij = second_invariant(τij)
            else
                # stress correction @ center
                Base.@nexprs 3 i -> begin
                    τ[i][I...] = dτij[i] + τij[i]
                end
                #τII_ij
            end
            #τII[I...] = τII_ij
#            η_vep[I...] = τII_ij * 0.5 * inv(second_invariant(εij))
            Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

        else
            Pr_c[I...] = zero(eltype(T))
            Base.@nexprs 3 i -> begin
                τ[i][I...] = zero(eltype(T))
            end

        end
    end

    return nothing
end


#=
@parallel_indices (I...) function update_stresses_center_vertexADOLD!(
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
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        #volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        volumev = (Kv == Inf) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(
            τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv
        )
#       τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
        τIIv_ij = av_clamped(τII, Ic...)

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv

        if is_pl && !iszero(τIIv_ij)  && Fv > 0
            τIIv_ijAD = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
            FvAD = τIIv_ijAD - Cv * cosϕv - Pv_ij * sinϕv
            # stress correction @ vertex
#            λv[I...] =
#               (1.0 - relλ) * λv[I...] +
#                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            λv[I...] =
                (1.0 - relλ) * λv[I...] +
                relλ * (FvAD / (ηv_ij * dτ_rv + η_regv + volumev))

#                λv[I...] = (FvAD / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ijAD
            εij_pl = λv[I...] * dQdτxy
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * εij_pl * dτ_rv
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
            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            volume = (K == Inf) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
#            εij_ve = @. εij + 0.5 * τij_o * _Gdt
#            εII_ve = GeoParams.second_invariant(εij_ve)
            # stress increments @ center
            dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
#            τII_ij = GeoParams.second_invariant(dτij .+ τij)
           τII_ij = τII[I...]
            # yield function @ center
            F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

#            τII_ij = 
            if is_pl && !iszero(τII_ij) && F > 0
                τII_ijAD = GeoParams.second_invariant(dτij .+ τij)
                FAD = τII_ijAD - C * cosϕ - Pr[I...] * sinϕ
                # stress correction @ center
#                λ[I...] =
#                    (1.0 - relλ) * λ[I...] +
#                    relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                λ[I...] =
                    (1.0 - relλ) * λ[I...] +
                    relλ * (FAD / (η[I...] * dτ_r + η_reg + volume))

#                λ[I...] = (FAD / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ijAD
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
#                setindex!.(ε_pl, εij_pl, I...)
#                τII_ij = GeoParams.second_invariant(τij)
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)
#                τII_ij
            end
#            τII[I...] = τII_ij
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

@parallel_indices (I...) function update_stresses_center_vertexADSensOLD!(
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
        Gv = Sens[4][I...]
        frv = av_clamped(Sens[5], Ic...)

        ## vertex
        phase = @inbounds phase_vertex[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase)
        #_Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        _Gvdt = inv(Gv * dt)
        #sinϕv = isinf(frv) ? 0.0 : sind(frv)
        #cosϕv = isinf(frv) ? 0.0 : cosd(frv)
        #sinϕv = (frv == Inf) ? 0.0 : sind(frv)
        #cosϕv = (frv == Inf) ? 0.0 : cosd(frv)

        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        #volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        volumev = (Kv == Inf) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        ηv_ij = av_clamped(η, Ic...)
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(
            τxyv[I...], τxyv_old[I...], ηv_ij, ε[3][I...], _Gvdt, dτ_rv)

      
#        τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
        τIIv_ij = av_clamped(τII, Ic...)

        # yield function @ center
        Fv = τIIv_ij - Cv * cosϕv - Pv_ij * sinϕv

        if is_pl && !iszero(τIIv_ij)  && Fv > 0
            τIIv_ijAD = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
            FvAD = τIIv_ijAD - Cv * cosϕv - Pv_ij * sinϕv
            # stress correction @ vertex
            #λv[I...] =
            #    (1.0 - relλ) * λv[I...] +
            #    relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
            λv[I...] = (1.0 - relλ) * λv[I...] + relλ * (FvAD / (ηv_ij * dτ_rv + η_regv + volumev))
            dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ijAD
            εij_pl = λv[I...] * dQdτxy
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * εij_pl * dτ_rv
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
            G     = @inbounds Sens[1][I...]
            fr    = @inbounds Sens[2][I...]
            #_Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
            _Gdt = inv(G * dt)
            #sinϕ = isinf(fr) ? 0.0 : sind(fr)
            #cosϕ = isinf(fr) ? 0.0 : cosd(fr)
            #sinϕ = (fr == Inf) ? 0.0 : sind(fr)
            #cosϕ = (fr == Inf) ? 0.0 : cosd(fr)

            is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
            K = fn_ratio(get_bulk_modulus, rheology, phase)
            #volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            volume = (K == Inf) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
            ηij = η[I...]
            dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)
            # cache strain rates for center calculations
            τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

            # visco-elastic strain rates @ center
            εij_ve = @. εij + 0.5 * τij_o * _Gdt
            εII_ve = GeoParams.second_invariant(εij_ve)
            # stress increments @ center
            dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)

#            τII_ij = GeoParams.second_invariant(dτij .+ τij)
            τII_ij = τII[I...]
            # yield function @ center
            F = τII_ij - C * cosϕ - Pr[I...] * sinϕ

#            τII_ij = 
            if is_pl && !iszero(τII_ij) && F > 0
                τII_ijAD = GeoParams.second_invariant(dτij .+ τij)
                FAD = τII_ijAD - C * cosϕ - Pr[I...] * sinϕ
                # stress correction @ center
                #λ[I...] =
                #    (1.0 - relλ) * λ[I...] +
                #    relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
                λ[I...] = (1.0 - relλ) * λ[I...] + relλ * (FAD / (η[I...] * dτ_r + η_reg + volume))
                dQdτij = @. 0.5 * (τij + dτij) / τII_ijAD
                εij_pl = λ[I...] .* dQdτij
                dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
                τij = dτij .+ τij
                setindex!.(τ, τij, I...)
#                setindex!.(ε_pl, εij_pl, I...)
#                τII_ij = GeoParams.second_invariant(τij)
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)

#              τII_ij
            end
#            τII[I...] = τII_ij
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


# Viscous
function compute_stress_incrementAD(τij::Real, τij_o::Real, ηij, εij::Real, _Gdt, dτ_r)
    dτij = dτ_r * fma(2.0 * ηij, εij, fma(-(τij - τij_o) * ηij, _Gdt, 0.0))
    return dτij
end

function compute_stress_incrementAD(
        τij::NTuple{N}, τij_o::NTuple{N}, ηij, εij::NTuple{N}, _Gdt, dτ_r
    ) where {N}
    dτij = ntuple(Val(N)) do i
        Base.@_inline_meta
        return dτ_r *
            fma(2.0 * ηij, εij[i], fma(-((τij[i] - τij_o[i])) * ηij, _Gdt, 0.0))
    end
    return dτij
end
=#