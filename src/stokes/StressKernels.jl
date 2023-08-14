# Viscous

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, εxx, εyy, εxy, η, θ_dτ
) where {T}
    av(A) = _av(A, i, j)

    denominator = inv(θ_dτ + 1.0)
    η_ij = η[i, j]

    # Normal components
    τxx[i, j] += (-τxx[i, j] + 2.0 * η_ij * εxx[i, j]) * denominator
    τyy[i, j] += (-τyy[i, j] + 2.0 * η_ij * εyy[i, j]) * denominator
    # Shear components
    if all((i, j) .≤ size(τxy))
        τxy[i + 1, j + 1] +=
            (-τxy[i + 1, j + 1] + 2.0 * @av(η) * εxy[i + 1, j + 1]) * denominator
    end

    return nothing
end

# Visco-elastic

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
) where {T}
    av(A) = _av(A, i, j)

    # Normal components
    _Gdt = inv(G[i, j] * dt)
    η_ij = η[i, j]
    denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
    τxx[i, j] +=
        (-(τxx[i, j] - τxx_o[i, j]) * η_ij * _Gdt - τxx[i, j] + 2.0 * η_ij * εxx[i, j]) *
        denominator
    τyy[i, j] +=
        (-(τyy[i, j] - τyy_o[i, j]) * η_ij * _Gdt - τyy[i, j] + 2.0 * η_ij * εyy[i, j]) *
        denominator

    # Shear components
    if all((i, j) .≤ size(τxy))
        av_η_ij = av(η)
        av_Gdt = inv(av(G) * dt)
        denominator = inv(θ_dτ + av_η_ij * av_Gdt + 1.0)
        τxy[i + 1, j + 1] +=
            (
                -(τxy[i + 1, j + 1] - τxy_o[i + 1, j + 1]) * av_η_ij * av_Gdt +
                2.0 * av_η_ij * εxy[i + 1, j + 1]
            ) * denominator
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_τ!(
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    τxx_o,
    τyy_o,
    τzz_o,
    τyz_o,
    τxz_o,
    τxy_o,
    εxx,
    εyy,
    εzz,
    εyz,
    εxz,
    εxy,
    η,
    G,
    dt,
    θ_dτ,
)
    harm_xy(A) = _harm_xyi(A, i, j, k)
    harm_xz(A) = _harm_xzi(A, i, j, k)
    harm_yz(A) = _harm_yzi(A, i, j, k)
    av_xy(A) = _av_xyi(A, i, j, k)
    av_xz(A) = _av_xzi(A, i, j, k)
    av_yz(A) = _av_yzi(A, i, j, k)
    get(x) = x[i, j, k]

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            _Gdt = inv(get(G) * dt)
            η_ij = get(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            # Compute τ_xx
            τxx[i, j, k] +=
                (
                    -(get(τxx) - get(τxx_o)) * η_ij * _Gdt - get(τxx) +
                    2.0 * η_ij * get(εxx)
                ) * denominator
            # Compute τ_yy
            τyy[i, j, k] +=
                (
                    -(get(τyy) - get(τyy_o)) * η_ij * _Gdt - get(τyy) +
                    2.0 * η_ij * get(εyy)
                ) * denominator
            # Compute τ_zz
            τzz[i, j, k] +=
                (
                    -(get(τzz) - get(τzz_o)) * η_ij * _Gdt - get(τzz) +
                    2.0 * η_ij * get(εzz)
                ) * denominator
        end
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            _Gdt = inv(harm_xy(G) * dt)
            η_ij = harm_xy(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxy[i, j, k] +=
                (
                    -(get(τxy) - get(τxy_o)) * η_ij * _Gdt - get(τxy) +
                    2.0 * η_ij * get(εxy)
                ) * denominator
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            _Gdt = inv(harm_xz(G) * dt)
            η_ij = harm_xz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxz[i, j, k] +=
                (
                    -(get(τxz) - get(τxz_o)) * η_ij * _Gdt - get(τxz) +
                    2.0 * η_ij * get(εxz)
                ) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            _Gdt = inv(harm_yz(G) * dt)
            η_ij = harm_yz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τyz[i, j, k] +=
                (
                    -(get(τyz) - get(τyz_o)) * η_ij * _Gdt - get(τyz) +
                    2.0 * η_ij * get(εyz)
                ) * denominator
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
    τyz, τxz, τxy, τyz_o, τxz_o, τxy_o, εyz, εxz, εxy, η, G, dt, θ_dτ
)
    harm_xy(A) = _harm_xyi(A, i, j, k)
    harm_xz(A) = _harm_xzi(A, i, j, k)
    harm_yz(A) = _harm_yzi(A, i, j, k)
    av_xy(A) = _av_xyi(A, i, j, k)
    av_xz(A) = _av_xzi(A, i, j, k)
    av_yz(A) = _av_yzi(A, i, j, k)
    get(x) = x[i, j, k]

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            _Gdt = inv(av_xy(G) * dt)
            η_ij = harm_xy(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxy[i, j, k] +=
                (
                    -(get(τxy) - get(τxy_o)) * η_ij * _Gdt - get(τxy) +
                    2.0 * η_ij * get(εxy)
                ) * denominator
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            _Gdt = inv(av_xz(G) * dt)
            η_ij = harm_xz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxz[i, j, k] +=
                (
                    -(get(τxz) - get(τxz_o)) * η_ij * _Gdt - get(τxz) +
                    2.0 * η_ij * get(εxz)
                ) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            _Gdt = inv(av_yz(G) * dt)
            η_ij = harm_yz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τyz[i, j, k] +=
                (
                    -(get(τyz) - get(τyz_o)) * η_ij * _Gdt - get(τyz) +
                    2.0 * η_ij * get(εyz)
                ) * denominator
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
    τyz, τxz, τxy, τyz_o, τxz_o, τxy_o, εyz, εxz, εxy, η, rheology, phase_ratios, dt, θ_dτ
)
    get(x) = x[i, j, k]
    harm_xy(A) = _harm_xyi(A, i, j, k)
    harm_xz(A) = _harm_xzi(A, i, j, k)
    harm_yz(A) = _harm_yzi(A, i, j, k)
    _eval(A, i, j, k) = fn_ratio(get_G, rheology, A[i, j, k])
    @inline Base.@propagate_inbounds function av_Gdt_xy(A)
        x = 0.0
        for ii in (i - 1):i, jj in (j - 1):j, kk in k:k
            x += _eval(A, ii, jj, kk)
        end
        return x * 0.25
    end
    @inline Base.@propagate_inbounds function av_Gdt_xz(A)
        x = 0.0
        for ii in (i - 1):i, jj in j:j, kk in (k - 1):k
            x += _eval(A, ii, jj, kk)
        end
        return x * 0.25
    end
    @inline Base.@propagate_inbounds function av_Gdt_yz(A)
        x = 0.0
        for ii in i:i, jj in (j - 1):j, kk in (k - 1):k
            x += _eval(A, ii, jj, kk)
        end
        return x * 0.25
    end

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            _Gdt = av_Gdt_xy(phase_ratios)
            η_ij = harm_xy(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxy[i, j, k] +=
                (
                    -(get(τxy) - get(τxy_o)) * η_ij * _Gdt - get(τxy) +
                    2.0 * η_ij * get(εxy)
                ) * denominator
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            _Gdt = av_Gdt_xz(phase_ratios)
            η_ij = harm_xz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τxz[i, j, k] +=
                (
                    -(get(τxz) - get(τxz_o)) * η_ij * _Gdt - get(τxz) +
                    2.0 * η_ij * get(εxz)
                ) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            _Gdt = av_Gdt_yz(phase_ratios)
            η_ij = harm_yz(η)
            denominator = inv(θ_dτ + η_ij * _Gdt + 1.0)
            τyz[i, j, k] +=
                (
                    -(get(τyz) - get(τyz_o)) * η_ij * _Gdt - get(τyz) +
                    2.0 * η_ij * get(εyz)
                ) * denominator
        end
    end
    return nothing
end

# Single phase visco-elasto-plastic flow

@parallel_indices (i, j) function compute_τ_nonlinear!(
    τxx::AbstractArray{T,2},
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    λ,
    rheology,
    dt,
    θ_dτ,
) where {T}
    I = i, j

    # numerics
    ηij = η[I...]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τxy
    τ_old = τxx_old, τyy_old, τxyv_old
    ε = εxx, εyy, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (i, j) function compute_τ_nonlinear!(
    τxx::AbstractArray{T,2},
    τyy,
    τxy,
    τII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    λ,
    phase_ratios::PhaseRatio,
    rheology,
    dt,
    θ_dτ,
) where {T}
    I = i, j

    # numerics
    ηij = @inbounds η[I...]
    phase = @inbounds phase_ratios[I...]
    G = fn_ratio(get_G, MatParam, phase)
    _Gdt = inv(G * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params_phase(rheology, phase)

    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τxy
    τ_old = τxx_old, τyy_old, τxyv_old
    ε = εxx, εyy, εxyv

    _compute_τ_nonlinear!(
        τ,
        τII,
        τ_old,
        ε,
        P,
        ηij,
        η_vep,
        phase_ratios,
        λ,
        dτ_r,
        _Gdt,
        plastic_parameters,
        I...,
    )

    return nothing
end

@parallel_indices (i, j, k) function compute_τ_nonlinear!(
    τxx::AbstractArray{T,3},
    τyy,
    τzz,
    τyz, # @ cell centers
    τxz, # @ cell centers
    τxy, # @ cell centers
    τII,
    τxx_old,
    τyy_old,
    τzz_old,
    τyzv_old, # @ vertices
    τxzv_old, # @ vertices
    τxyv_old, # @ vertices
    εxx,
    εyy,
    εzz,
    εyzv, # @ vertices
    εxzv, # @ vertices
    εxyv, # @ vertices
    P,
    η,
    η_vep,
    λ,
    rheology,
    dt,
    θ_dτ,
) where {T}
    I = i, j, k

    # numerics
    ηij = η[I...]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_old = τxx_old, τyy_old, τzz_old, τyzv_old, τxzv_old, τxyv_old
    ε = εxx, εyy, εzz, εyzv, εxzv, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (i, j, k) function compute_τ_nonlinear!(
    τxx::AbstractArray{T,3},
    τyy,
    τzz,
    τyz, # @ cell centers
    τxz, # @ cell centers
    τxy, # @ cell centers
    τII,
    τxx_old,
    τyy_old,
    τzz_old,
    τyzv_old, # @ vertices
    τxzv_old, # @ vertices
    τxyv_old, # @ vertices
    εxx,
    εyy,
    εzz,
    εyzv, # @ vertices
    εxzv, # @ vertices
    εxyv, # @ vertices
    P,
    η,
    η_vep,
    λ,
    phase_center,
    rheology,
    dt,
    θ_dτ,
) where {T}
    I = i, j, k

    # numerics
    ηij = @inbounds η[I...]
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_G, rheology, phase) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params_phase(rheology, phase)
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_old = τxx_old, τyy_old, τzz_old, τyzv_old, τxzv_old, τxyv_old
    ε = εxx, εyy, εzz, εyzv, εxzv, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )

    return nothing
end
