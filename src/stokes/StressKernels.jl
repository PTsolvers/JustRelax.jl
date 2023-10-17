# Viscous

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, εxx, εyy, εxy, η, θ_dτ
) where {T}
    @inline av(A) = _av_a(A, i, j)

    denominator = inv(θ_dτ + 1.0)
    η_ij = η[i, j]

    # Normal components
    τxx[i, j] += (-τxx[i, j] + 2.0 * η_ij * εxx[i, j]) * denominator
    τyy[i, j] += (-τyy[i, j] + 2.0 * η_ij * εyy[i, j]) * denominator
    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        τxy[i + 1, j + 1] +=
            (-τxy[i + 1, j + 1] + 2.0 * @av(η) * εxy[i + 1, j + 1]) * denominator
    end

    return nothing
end

# Visco-elastic

@parallel_indices (i, j) function compute_τ!(
    τxx::AbstractArray{T,2}, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
) where {T}
    @inline av(A) = _av_a(A, i, j)

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
    if all((i, j) .< size(τxy) .- 1)
        av_η_ij = av(η)
        _av_Gdt = inv(av(G) * dt)
        denominator = inv(θ_dτ + av_η_ij * _av_Gdt + 1.0)
        τxy[i + 1, j + 1] +=
            (
                -(τxy[i + 1, j + 1] - τxy_o[i + 1, j + 1]) * av_η_ij * _av_Gdt +
                2.0 * av_η_ij * εxy[i + 1, j + 1]
            ) * denominator
    end

    return nothing
end

@parallel_indices (i, j) function compute_τ_vertex!(
    τxy::AbstractArray{T,2}, εxy, η, θ_dτ
) where {T}
    @inline av(A) = _av_a(A, i, j)

    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        I = i + 1, j + 1
        av_η_ij = av(η)
        denominator = inv(θ_dτ + 1.0)

        τxy[I...] += (-τxy[I...] + 2.0 * av_η_ij * εxy[I...]) * denominator
    end

    return nothing
end

@parallel_indices (i, j) function compute_τ_vertex!(
    τxy::AbstractArray{T,2}, εxy, η, θ_dτ
) where {T}
    @inline av(A) = _av_a(A, i, j)

    # Shear components
    if all((i, j) .< size(τxy) .- 1)
        I = i + 1, j + 1
        av_η_ij = av(η)
        denominator = inv(θ_dτ + 1.0)

        τxy[I...] += (-τxy[I...] + 2.0 * av_η_ij * εxy[I...]) * denominator
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
    τyz, τxz, τxy, εyz, εxz, εxy, ηvep, θ_dτ
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
            η_ij = harm_xy(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τxy[i, j, k] += (-get(τxy) + 2.0 * η_ij * get(εxy)) * denominator
            denominator = inv(θ_dτ + 1.0)
            τxy[i, j, k] += (-get(τxy) + 2.0 * η_ij * get(εxy)) * denominator
        end

        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            η_ij = harm_xz(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τxz[i, j, k] += (-get(τxz) + 2.0 * η_ij * get(εxz)) * denominator
            denominator = inv(θ_dτ + 1.0)
            τxz[i, j, k] += (-get(τxz) + 2.0 * η_ij * get(εxz)) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            η_ij = harm_yz(ηvep)
            denominator = inv(θ_dτ + 1.0)
            τyz[i, j, k] += (-get(τyz) + 2.0 * η_ij * get(εyz)) * denominator
            denominator = inv(θ_dτ + 1.0)
            τyz[i, j, k] += (-get(τyz) + 2.0 * η_ij * get(εyz)) * denominator
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
    τyz, τxz, τxy, εyz, εxz, εxy, η, θ_dτ
)
    I = i, j, k
    harm_xy(A) = _harm_xyi(A, I...)
    harm_xz(A) = _harm_xzi(A, I...)
    harm_yz(A) = _harm_yzi(A, I...)
    av_xy(A) = _av_xyi(A, I...)
    av_xz(A) = _av_xzi(A, I...)
    av_yz(A) = _av_yzi(A, I...)

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            η_ij = av_xy(η)
            denominator = inv(θ_dτ + 1.0)
            τxy[I...] += (-τxy[I...] + 2.0 * η_ij * εxy[I...]) * denominator
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            η_ij = av_xz(η)
            denominator = inv(θ_dτ + 1.0)
            τxz[I...] += (-τxz[I...] + 2.0 * η_ij * εxz[I...]) * denominator
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            η_ij = av_yz(η)
            denominator = inv(θ_dτ + 1.0)
            τyz[I...] += (-τyz[I...] + 2.0 * η_ij * εyz[I...]) * denominator
        end
    end
    return nothing
end

# Single phase visco-elasto-plastic flow

@parallel_indices (I...) function compute_τ_nonlinear!(
    τ,     # shear @ centers
    τII,
    τ_old, # shear @ centers
    ε,     # shear @ vertices
    P,
    η,
    η_vep,
    λ,
    rheology,
    dt,
    θ_dτ,
)

    # numerics
    ηij = η[I...]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (I...) function compute_τ_nonlinear!(
    τ,     # @ cell centers
    τII,
    τ_old, # @ cell centers
    ε,     # @ vertices
    P,
    θ,
    η,
    η_vep,
    λ,
    phase_center,
    rheology,
    dt,
    θ_dτ,
)
    # numerics
    ηij = @inbounds η[I...]
    phase = @inbounds phase_center[I...]
    _Gdt = inv(fn_ratio(get_G, rheology, phase) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, phase)
    # plastic volumetric change K*dt*sinϕ*sinψ
    K = fn_ratio(get_bulkmodulus, rheology, phase)
    volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ
    plastic_parameters = (; is_pl, C, sinϕ, cosϕ, η_reg, volume)

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, I...
    )
    θ[I...] = P[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)

    return nothing
end

## Stress invariants 
@parallel_indices (I...) function tensor_invariant_center!(
    II, tensor::NTuple{N,T}
) where {N,T}
    II[I...] = second_invariant_staggered(getindex(tensor, I...)...)
    return nothing
end

@parallel_indices (I...) function tensor_invariant!(II, xx, yy, xyv)
    # convinience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        τ = xx[I...], yy[I...], gather(xyv)
        II[I...] = second_invariant_staggered(τ...)
    end

    return nothing
end

@parallel_indices (I...) function tensor_invariant!(II, xx, yy, zz, yz, xz, xy)

    # convinience closure
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        τ = xx[I...], yy[I...], zz[I...], gather_yz(yz), gather_xz(xz), gather_xy(xy)
        II[I...] = second_invariant_staggered(τ...)
    end

    return nothing
end
