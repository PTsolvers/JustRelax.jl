# 2D kernel
@parallel_indices (I...) function update_stresses_center_vertex!(
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
                τII[I...] = GeoParams.second_invariant(τij)
                Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
                η_vep[I...] = 0.5 * τII_ij / εII_ve
            else
                # stress correction @ center
                setindex!.(τ, dτij .+ τij, I...)
                η_vep[I...] = ηij
                τII[I...] = τII_ij
            end

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

# 3D kernel
@parallel_indices (I...) function update_stresses_center_vertex!(
        ε::NTuple{6},         # normal components @ centers; shear components @ vertices
        ε_pl::NTuple{6},      # whole Voigt tensor @ centers
        EII,                  # accumulated plastic strain rate @ centers
        τ::NTuple{6},         # whole Voigt tensor @ centers
        τshear_v::NTuple{3},  # shear tensor components @ vertices
        τ_o::NTuple{6},
        τshear_ov::NTuple{3}, # shear tensor components @ vertices
        Pr,
        Pr_c,
        η,
        λ,
        λv::NTuple{3},
        τII,
        η_vep,
        relλ,
        dt,
        θ_dτ,
        rheology,
        phase_center,
        phase_vertex,
        phase_xy,
        phase_yz,
        phase_xz,
        ϕ::JustRelax.RockRatio,
    )
    τyzv, τxzv, τxyv = τshear_v
    τyzv_old, τxzv_old, τxyv_old = τshear_ov

    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    ## yz
    if all(I .≤ size(ε[4])) && isvalid_yz(ϕ, I...)
        # interpolate to ith vertex
        ηv_ij = av_clamped_yz(η, Ic...)
        Pv_ij = av_clamped_yz(Pr, Ic...)
        EIIv_ij = av_clamped_yz(EII, Ic...)
        εxxv_ij = av_clamped_yz(ε[1], Ic...)
        εyyv_ij = av_clamped_yz(ε[2], Ic...)
        εzzv_ij = av_clamped_yz(ε[3], Ic...)
        εyzv_ij = ε[4][I...]
        εxzv_ij = av_clamped_yz_y(ε[5], Ic...)
        εxyv_ij = av_clamped_yz_z(ε[6], Ic...)

        τxxv_ij = av_clamped_yz(τ[1], Ic...)
        τyyv_ij = av_clamped_yz(τ[2], Ic...)
        τzzv_ij = av_clamped_yz(τ[3], Ic...)
        τyzv_ij = τyzv[I...]
        τxzv_ij = av_clamped_yz_y(τxzv, Ic...)
        τxyv_ij = av_clamped_yz_z(τxyv, Ic...)

        τxxv_old_ij = av_clamped_yz(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_yz(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_yz(τ_o[3], Ic...)
        τyzv_old_ij = τyzv_old[I...]
        τxzv_old_ij = av_clamped_yz_y(τxzv_old, Ic...)
        τxyv_old_ij = av_clamped_yz_z(τxyv_old, Ic...)

        # vertex parameters
        phase = @inbounds phase_yz[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)

        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[1][I...] =
                (1.0 - relλ) * λv[1][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτyz = 0.5 * (τyzv_ij + dτyzv) / τIIv_ij
            τyzv[I...] += dτyzv - 2.0 * ηv_ij * λv[1][I...] * dQdτyz * dτ_rv
        else
            # stress correction @ vertex
            τyzv[I...] += dτyzv
        end
    end

    ## xz
    if all(I .≤ size(ε[5])) && isvalid_xz(ϕ, I...)
        # interpolate to ith vertex
        ηv_ij = av_clamped_xz(η, Ic...)
        EIIv_ij = av_clamped_xz(EII, Ic...)
        Pv_ij = av_clamped_xz(Pr, Ic...)
        εxxv_ij = av_clamped_xz(ε[1], Ic...)
        εyyv_ij = av_clamped_xz(ε[2], Ic...)
        εzzv_ij = av_clamped_xz(ε[3], Ic...)
        εyzv_ij = av_clamped_xz_x(ε[4], Ic...)
        εxzv_ij = ε[5][I...]
        εxyv_ij = av_clamped_xz_z(ε[6], Ic...)
        τxxv_ij = av_clamped_xz(τ[1], Ic...)
        τyyv_ij = av_clamped_xz(τ[2], Ic...)
        τzzv_ij = av_clamped_xz(τ[3], Ic...)
        τyzv_ij = av_clamped_xz_x(τyzv, Ic...)
        τxzv_ij = τxzv[I...]
        τxyv_ij = av_clamped_xz_z(τxyv, Ic...)
        τxxv_old_ij = av_clamped_xz(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_xz(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_xz(τ_o[3], Ic...)
        τyzv_old_ij = av_clamped_xz_x(τyzv_old, Ic...)
        τxzv_old_ij = τxzv_old[I...]
        τxyv_old_ij = av_clamped_xz_z(τxyv_old, Ic...)

        # vertex parameters
        phase = @inbounds phase_xz[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)

        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[2][I...] =
                (1.0 - relλ) * λv[2][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxz = 0.5 * (τxzv_ij + dτxzv) / τIIv_ij
            τxzv[I...] += dτxzv - 2.0 * ηv_ij * λv[2][I...] * dQdτxz * dτ_rv
        else
            # stress correction @ vertex
            τxzv[I...] += dτxzv
        end
    end

    ## xy
    if all(I .≤ size(ε[6])) && isvalid_xy(ϕ, I...)
        # interpolate to ith vertex
        ηv_ij = av_clamped_xy(η, Ic...)
        EIIv_ij = av_clamped_xy(EII, Ic...)
        Pv_ij = av_clamped_xy(Pr, Ic...)
        εxxv_ij = av_clamped_xy(ε[1], Ic...)
        εyyv_ij = av_clamped_xy(ε[2], Ic...)
        εzzv_ij = av_clamped_xy(ε[3], Ic...)
        εyzv_ij = av_clamped_xy_x(ε[4], Ic...)
        εxzv_ij = av_clamped_xy_y(ε[5], Ic...)
        εxyv_ij = ε[6][I...]

        τxxv_ij = av_clamped_xy(τ[1], Ic...)
        τyyv_ij = av_clamped_xy(τ[2], Ic...)
        τzzv_ij = av_clamped_xy(τ[3], Ic...)
        τyzv_ij = av_clamped_xy_x(τyzv, Ic...)
        τxzv_ij = av_clamped_xy_y(τxzv, Ic...)
        τxyv_ij = τxyv[I...]

        τxxv_old_ij = av_clamped_xy(τ_o[1], Ic...)
        τyyv_old_ij = av_clamped_xy(τ_o[2], Ic...)
        τzzv_old_ij = av_clamped_xy(τ_o[3], Ic...)
        τyzv_old_ij = av_clamped_xy_x(τyzv_old, Ic...)
        τxzv_old_ij = av_clamped_xy_y(τxzv_old, Ic...)
        τxyv_old_ij = τxyv_old[I...]

        # vertex parameters
        phase = @inbounds phase_xy[I...]
        is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(
            rheology, EIIv_ij, phase
        )
        _Gvdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        Kv = fn_ratio(get_bulk_modulus, rheology, phase)
        volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
        dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + 1.0)

        # stress increments @ vertex
        dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, εxxv_ij, _Gvdt, dτ_rv)
        dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, εyyv_ij, _Gvdt, dτ_rv)
        dτzzv = compute_stress_increment(τzzv_ij, τzzv_old_ij, ηv_ij, εzzv_ij, _Gvdt, dτ_rv)
        dτyzv = compute_stress_increment(τyzv_ij, τyzv_old_ij, ηv_ij, εyzv_ij, _Gvdt, dτ_rv)
        dτxzv = compute_stress_increment(τxzv_ij, τxzv_old_ij, ηv_ij, εxzv_ij, _Gvdt, dτ_rv)
        dτxyv = compute_stress_increment(τxyv_ij, τxyv_old_ij, ηv_ij, εxyv_ij, _Gvdt, dτ_rv)
        dτijv = dτxxv, dτyyv, dτzzv, dτyzv, dτxzv, dτxyv
        τijv = τxxv_ij, τyyv_ij, τzzv_ij, τyzv_ij, τxzv_ij, τxyv_ij
        τIIv_ij = second_invariant(τijv .+ dτijv)

        # yield function @ vertex
        Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
        if is_pl && !iszero(τIIv_ij) && Fv > 0
            # stress correction @ vertex
            λv[3][I...] =
                (1.0 - relλ) * λv[3][I...] +
                relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))

            dQdτxy = 0.5 * (τxyv_ij + dτxyv) / τIIv_ij
            τxyv[I...] += dτxyv - 2.0 * ηv_ij * λv[3][I...] * dQdτxy * dτ_rv
        else
            # stress correction @ vertex
            τxyv[I...] += dτxyv
        end
    end

    ## center
    if all(I .≤ ni) && isvalid_c(ϕ, I...)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = inv(θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = second_invariant(εij_ve)
        # stress increments @ center
        dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij + 2.0 * ηij * εij) * dτ_r
        τII_ij = second_invariant(dτij .+ τij)
        # yield function @ center
        F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

        if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            τII[I...] = second_invariant(τij)
            Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
            η_vep[I...] = 0.5 * τII_ij / εII_ve
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            η_vep[I...] = ηij
            τII[I...] = τII_ij
        end

        Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end

    return nothing
end
