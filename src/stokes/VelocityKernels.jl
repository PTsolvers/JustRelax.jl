## DIVERGENCE

@parallel_indices (I...) function compute_∇V!(∇V::AbstractArray, V::NTuple, _di::NTuple)
    @inbounds ∇V[I...] = div(V..., _di..., I...)
    return nothing
end

## DEVIATORIC STRAIN RATE TENSOR

@parallel_indices (i, j) function compute_strain_rate!(
        εxx::AbstractArray{T, 2}, εyy, εxy, ∇V, Vx, Vy, _dx, _dy
    ) where {T}

    @inbounds begin
        Vx1 = Vx[i, j]
        Vx2 = Vx[i, j + 1]
        Vy1 = Vy[i, j]
        Vy2 = Vy[i + 1, j]

        if all((i, j) .≤ size(εxx))
            Vx3 = Vx[i + 1, j + 1]
            Vy3 = Vy[i + 1, j + 1]

            ∇V_ij = ∇V[i, j] / 3
            εxx[i, j] = (Vx3 - Vx2) * _dx - ∇V_ij
            εyy[i, j] = (Vy3 - Vy2) * _dy - ∇V_ij
        end

        εxy[i, j] = 0.5 * ((Vx2 - Vx1) * _dy + (Vy2 - Vy1) * _dx)
    end
    return nothing
end

@parallel_indices (i, j) function compute_strain_rate_from_increment!(
        εxx::AbstractArray{T, 2}, εyy, εxy, Δεxx, Δεyy, Δεxy, _dt
    ) where {T}

    if all((i, j) .≤ size(εxx))
        εxx[i, j] = Δεxx[i, j] * _dt
        εyy[i, j] = Δεyy[i, j] * _dt
    end
    εxy[i, j] = Δεxy[i, j] * _dt

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
        ∇V::AbstractArray{T, 3}, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
    ) where {T}
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)

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
        Vx::AbstractArray{T, 2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy
    ) where {T}
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        @inbounds Vx[i + 1, j + 1] +=
            (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
    end
    if all((i, j) .< size(Vy) .- 1)
        @inbounds Vy[i + 1, j + 1] +=
            (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy)) * ηdτ / av_ya(ητ)
    end
    return nothing
end

# with free surface stabilization
@parallel_indices (i, j) function compute_V!(
        Vx::AbstractArray{T, 2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy, dt
    ) where {T}
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)

    if all((i, j) .< size(Vx) .- 1)
        @inbounds Vx[i + 1, j + 1] +=
            (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
    end

    @inbounds if all((i, j) .< size(Vy) .- 1)
        θ = 1.0
        # Interpolated Vx into Vy node (includes density gradient)
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        j_N = min(j + 1, ny)
        ρg_S = ρgy[i, j]
        ρg_N = ρgy[i, j_N]
        # Spatial derivatives
        ∂ρg∂y = (ρg_N - ρg_S) * _dy
        # correction term
        ρg_correction = Vyᵢⱼ * ∂ρg∂y * θ * dt

        Vy[i + 1, j + 1] +=
            (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy) + ρg_correction) * ηdτ /
            av_ya(ητ)
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_V!(
        Vx::AbstractArray{T, 3},
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
    Base.@propagate_inbounds @inline harm_x(A) = _harm_x(A, i, j, k)
    Base.@propagate_inbounds @inline harm_y(A) = _harm_y(A, i, j, k)
    Base.@propagate_inbounds @inline harm_z(A) = _harm_z(A, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)

    @inbounds begin
        if all((i, j, k) .≤ size(Rx))
            Rx_ijk =
                Rx[i, j, k] =
                d_xa(τxx) +
                _dy * (τxy[i + 1, j + 1, k] - τxy[i + 1, j, k]) +
                _dz * (τxz[i + 1, j, k + 1] - τxz[i + 1, j, k]) - d_xa(P) - av_x(fx)
            Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
        end
        if all((i, j, k) .≤ size(Ry))
            Ry_ijk =
                Ry[i, j, k] =
                _dx * (τxy[i + 1, j + 1, k] - τxy[i, j + 1, k]) +
                _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
                _dz * (τyz[i, j + 1, k + 1] - τyz[i, j + 1, k]) - d_ya(P) - av_y(fy)
            Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
        end
        if all((i, j, k) .≤ size(Rz))
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
        Rx::AbstractArray{T, 2}, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

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
        Rx::AbstractArray{T, 2}, Ry, Vx, Vy, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, dt
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    if all((i, j) .≤ size(Rx))
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
    end

    @inbounds if all((i, j) .≤ size(Ry))
        θ = 1.0
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        j_N = min(j + 1, ny)
        ρg_S = ρgy[i, j]
        ρg_N = ρgy[i, j_N]
        # Spatial derivatives
        ∂ρg∂y = (ρg_N - ρg_S) * _dy
        # correction term
        ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt

        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy) + ρg_correction
    end

    return nothing
end
