@parallel_indices (I...) function compute_∇V!(
    ∇V::AbstractArray{T,N}, V::NTuple{N}, ϕ::JustRelax.RockRatio, _di::NTuple{N}
) where {T,N}
    if isvalid_c(ϕ, I...)
        @inbounds ∇V[I...] = div(V..., _di..., I...)
    else
        @inbounds ∇V[I...] = zero(T)
    end
    return nothing
end

@parallel_indices (i, j) function compute_strain_rate!(
    εxx::AbstractArray{T,2}, εyy, εxy, ∇V, Vx, Vy, ϕ::JustRelax.RockRatio, _dx, _dy
) where {T}
    @inline d_xi(A) = _d_xi(A, _dx, i, j)
    @inline d_yi(A) = _d_yi(A, _dy, i, j)
    @inline d_xa(A) = _d_xa(A, _dx, i, j)
    @inline d_ya(A) = _d_ya(A, _dy, i, j)
    @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)

    if all((i, j) .≤ size(εxx))
        if isvalid_c(ϕ, i, j)
            ∇V_ij = ∇V[i, j] / 3
            εxx[i, j] = d_xi(Vx) - ∇V_ij
            εyy[i, j] = d_yi(Vy) - ∇V_ij
        else
            εxx[i, j] = zero(T)
            εyy[i, j] = zero(T)
        end
        # εxx[i, j] = (Vx[i + 1, j + 1] * ϕ.Vx[i + 1, j] - Vx[i, j + 1] * ϕ.Vx[i, j]) * _dx - ∇V_ij
        # εyy[i, j] = (Vy[i + 1, j + 1] * ϕ.Vy[i, j + 1] - Vy[i + 1, j] * ϕ.Vy[i, j]) * _dy - ∇V_ij
    end

    εxy[i, j] = if isvalid_v(ϕ, i, j)
        0.5 * (
            d_ya(Vx) + 
            d_xa(Vy)
        )
    else
        zero(T)
    end

    # εxy[i, j] =  0.5 * (
    #         d_ya(Vx) + 
    #         d_xa(Vy)
    #     )
    # vy_mask_left  = ϕ.Vy[max(i - 1, 1), j]
    # vy_mask_right = ϕ.Vy[min(i + 1, size(ϕ.Vy, 1)), j]
    
    # vx_mask_bot = ϕ.Vx[i, max(j - 1, 1)]
    # vx_mask_top = ϕ.Vx[i, min(j + 1, size(ϕ.Vx, 2))]
    
    # εxy[i, j] = 0.5 * (
    #     (Vx[i, j+1] * vx_mask_top - Vx[i, j] * vx_mask_bot) * _dy + 
    #     (Vy[i+1, j] * vy_mask_right - Vy[i, j] * vy_mask_left) * _dx
    # )

    return nothing
end

@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2},
    Vy,
    Rx,
    Ry,
    P,
    τxx,
    τyy,
    τxy,
    ηdτ,
    ρgx,
    ρgy,
    ητ,
    ϕ::JustRelax.RockRatio,
    _dx,
    _dy,
) where {T}
    d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
    av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        if isvalid_vx(ϕ, i + 1, j)
            Rx[i, j] =
                R_Vx = (
                    -d_xa(P, ϕ.center) + d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) -
                    av_xa(ρgx, ϕ.center)
                )
            Vx[i + 1, j + 1] += R_Vx * ηdτ / av_xa(ητ)
        else
            Rx[i, j] = zero(T)
            Vx[i + 1, j + 1] = zero(T)
        end
    end

    if all((i, j) .< size(Vy) .- 1)
        if isvalid_vy(ϕ, i, j + 1)
            Ry[i, j] =
                R_Vy =
                    -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
                    av_ya(ρgy, ϕ.center)
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end
