## DIVERGENCE

@parallel_indices (i, j) function compute_∇V!(
    ∇V::AbstractArray{T,2}, Vx, Vy, _dx, _dy
) where {T}
    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)

    ∇V[i, j] = d_xi(Vx) + d_yi(Vy)

    return nothing
end

@parallel_indices (i, j, k) function compute_∇V!(
    ∇V::AbstractArray{T,3}, Vx, Vy, Vz, _dx, _dy, _dz
) where {T}
    d_xi(A) = _d_xi(A, i, j, k, _dx)
    d_yi(A) = _d_yi(A, i, j, k, _dy)
    d_zi(A) = _d_zi(A, i, j, k, _dz)

    @inbounds ∇V[i, j, k] = d_xi(Vx) + d_yi(Vy) + d_zi(Vz)
    return nothing
end

## DEVIATORIC STRAIN RATE TENSOR

@parallel_indices (i, j) function compute_strain_rate!(
    εxx::AbstractArray{T,2}, εyy, εxy, ∇V, Vx, Vy, _dx, _dy
) where {T}
    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)

    if all((i, j) .≤ size(εxx))
        ∇V_ij = ∇V[i, j] / 3.0
        εxx[i, j] = d_xi(Vx) - ∇V_ij
        εyy[i, j] = d_yi(Vy) - ∇V_ij
    end
    εxy[i, j] = 0.5 * (d_ya(Vx) + d_xa(Vy))

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
    ∇V::AbstractArray{T,3}, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
) where {T}
    d_xi(A) = _d_xi(A, i, j, k, _dx)
    d_yi(A) = _d_yi(A, i, j, k, _dy)
    d_zi(A) = _d_zi(A, i, j, k, _dz)

    @inbounds begin
        # normal components are all located @ cell centers
        if all((i, j, k) .≤ size(εxx))
            ∇Vijk = ∇V[i, j, k] * inv(3)
            # Compute ε_xx
            εxx[i, j, k] = d_xi(Vx) - ∇Vijk
            # Compute ε_yy
            εyy[i, j, k] = d_yi(Vy) - ∇Vijk
            # Compute ε_zz
            εzz[i, j, k] = d_zi(Vz) - ∇Vijk
        end
        # Compute ε_yz
        if all((i, j, k) .≤ size(εyz))
            εyz[i, j, k] =
                0.5 * (
                    _dz * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
                )
        end
        # Compute ε_xz
        if all((i, j, k) .≤ size(εxz))
            εxz[i, j, k] =
                0.5 * (
                    _dz * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
                )
        end
        # Compute ε_xy
        if all((i, j, k) .≤ size(εxy))
            εxy[i, j, k] =
                0.5 * (
                    _dy * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
                )
        end
    end
    return nothing
end

## VELOCITY 

@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy
) where {T}
    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        Vx[i + 1, j + 1] +=
            (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
    end
    if all((i, j) .< size(Vy) .- 1)
        Vy[i + 1, j + 1] +=
            (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy)) * ηdτ / av_ya(ητ)
    end
    return nothing
end

# with free surface stabilization
@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy, _dt
) where {T}
    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        Vx[i + 1, j + 1] +=
            (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx) + Vx[i + 1, j + 1] * d_xa(ρgy) * _dt) * ηdτ / av_xa(ητ)
    end
    if all((i, j) .< size(Vy) .- 1)
        Vy[i + 1, j + 1] +=
            (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy) + Vy[i + 1, j + 1] * d_ya(ρgy) * _dt) * ηdτ / av_ya(ητ)
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_V!(
    Vx::AbstractArray{T,3},
    Vy,
    Vz,
    Rx,
    Ry,
    Rz,
    P,
    fx,
    fy,
    fz,
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    ητ,
    ηdτ,
    _dx,
    _dy,
    _dz,
) where {T}
    harm_x(A) = _harm_x(A, i, j, k)
    harm_y(A) = _harm_y(A, i, j, k)
    harm_z(A) = _harm_z(A, i, j, k)
    av_x(A) = _av_x(A, i, j, k)
    av_y(A) = _av_y(A, i, j, k)
    av_z(A) = _av_z(A, i, j, k)
    d_xa(A) = _d_xa(A, i, j, k, _dx)
    d_ya(A) = _d_ya(A, i, j, k, _dy)
    d_za(A) = _d_za(A, i, j, k, _dz)

    @inbounds begin
        if all((i, j, k) .< size(Vx) .- 1)
            Rx_ijk =
                Rx[i, j, k] =
                    d_xa(τxx) +
                    _dy * (τxy[i + 1, j + 1, k] - τxy[i + 1, j, k]) +
                    _dz * (τxz[i + 1, j, k + 1] - τxz[i + 1, j, k]) - d_xa(P) - av_x(fx)
            Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
        end
        if all((i, j, k) .< size(Vy) .- 1)
            Ry_ijk = Ry[i, j, k]
            _dx * (τxy[i + 1, j + 1, k] - τxy[i, j + 1, k]) +
            _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
            _dz * (τyz[i, j + 1, k + 1] - τyz[i, j + 1, k]) - d_ya(P) - av_y(fy)
            Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
        end
        if all((i, j, k) .< size(Vz) .- 1)
            Rz_ijk =
                Rz[i, j, k] =
                    _dx * (τxz[i + 1, j, k + 1] - τxz[i, j, k + 1]) +
                    _dy * (τyz[i, j + 1, k + 1] - τyz[i, j, k + 1]) +
                    d_za(τzz) - d_za(P) - av_z(fz)
            Vz[i + 1, j + 1, k + 1] += Rz_ijk * ηdτ / av_z(ητ)
        end
    end

    return nothing
end

## RESIDUALS

@parallel_indices (i, j) function compute_Res!(
    Rx::AbstractArray{T,2}, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
) where {T}
    @inline d_xa(A) = _d_xa(A, i, j, _dx)
    @inline d_ya(A) = _d_ya(A, i, j, _dy)
    @inline d_xi(A) = _d_xi(A, i, j, _dx)
    @inline d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        end
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_Res!(
    Rx::AbstractArray{T,2}, Ry, Vx, Vy, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, _dt
) where {T}
    @inline d_xa(A) = _d_xa(A, i, j, _dx)
    @inline d_ya(A) = _d_ya(A, i, j, _dy)
    @inline d_xi(A) = _d_xi(A, i, j, _dx)
    @inline d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx) + Vx[i + 1, j + 1] * d_xa(ρgy) * _dt
        end
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy) + Vy[i + 1, j + 1] * d_ya(ρgy) * _dt
        end
    end
    return nothing
end