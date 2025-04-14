@parallel_indices (I...) function compute_∇V!(
        ∇V::AbstractArray{T, N}, V::NTuple{N}, ϕ::JustRelax.RockRatio, _di::NTuple{N}
    ) where {T, N}
    if isvalid_c(ϕ, I...)
        @inbounds ∇V[I...] = div(V..., _di..., I...)
    else
        @inbounds ∇V[I...] = zero(T)
    end
    return nothing
end

@parallel_indices (i, j) function compute_strain_rate!(
        εxx::AbstractArray{T, 2}, εyy, εxy, ∇V, Vx, Vy, ϕ::JustRelax.RockRatio, _dx, _dy
    ) where {T}
    @inline d_xi(A) = _d_xi(A, _dx, i, j)
    @inline d_yi(A) = _d_yi(A, _dy, i, j)
    @inline d_xa(A) = _d_xa(A, _dx, i, j)
    @inline d_ya(A) = _d_ya(A, _dy, i, j)

    if all((i, j) .≤ size(εxx))
        if isvalid_c(ϕ, i, j)
            ∇V_ij = ∇V[i, j] / 3
            εxx[i, j] = d_xi(Vx) - ∇V_ij
            εyy[i, j] = d_yi(Vy) - ∇V_ij
        else
            εxx[i, j] = zero(T)
            εyy[i, j] = zero(T)
        end
    end

    εxy[i, j] = if isvalid_v(ϕ, i, j)
        0.5 * (d_ya(Vx) + d_xa(Vy))
    else
        zero(T)
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
        ∇V::AbstractArray{T, 3},
        εxx,
        εyy,
        εzz,
        εyz,
        εxz,
        εxy,
        Vx,
        Vy,
        Vz,
        ϕ::JustRelax.RockRatio,
        _dx,
        _dy,
        _dz,
    ) where {T}
    d_xi(A) = _d_xi(A, _dx, i, j, k)
    d_yi(A) = _d_yi(A, _dy, i, j, k)
    d_zi(A) = _d_zi(A, _dz, i, j, k)

    @inbounds begin
        # normal components are all located @ cell centers
        if all((i, j, k) .≤ size(εxx))
            if isvalid_c(ϕ, i, j, k)
                ∇Vijk = ∇V[i, j, k] * inv(3)
                # Compute ε_xx
                εxx[i, j, k] = d_xi(Vx) - ∇Vijk
                # Compute ε_yy
                εyy[i, j, k] = d_yi(Vy) - ∇Vijk
                # Compute ε_zz
                εzz[i, j, k] = d_zi(Vz) - ∇Vijk
            end
        end
        # Compute ε_yz
        if all((i, j, k) .≤ size(εyz)) && isvalid_yz(ϕ, i, j, k)
            εyz[i, j, k] =
                0.5 * (
                _dz * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
            )
        end
        # Compute ε_xz
        if all((i, j, k) .≤ size(εxz)) && isvalid_xz(ϕ, i, j, k)
            εxz[i, j, k] =
                0.5 * (
                _dz * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
            )
        end
        # Compute ε_xy
        if all((i, j, k) .≤ size(εxy)) && isvalid_xy(ϕ, i, j, k)
            εxy[i, j, k] =
                0.5 * (
                _dy * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
            )
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_V!(
        Vx::AbstractArray{T, 2},
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

@parallel_indices (i, j) function compute_Vx!(
        Vx::AbstractArray{T, 2}, Rx, P, τxx, τxy, ηdτ, ρgx, ητ, ϕ::JustRelax.RockRatio, _dx, _dy
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

    return nothing
end

@parallel_indices (i, j) function compute_Vy!(
        Vy::AbstractArray{T, 2},
        Vx_on_Vy,
        Ry,
        P,
        τyy,
        τxy,
        ηdτ,
        ρgy,
        ητ,
        ϕ::JustRelax.RockRatio,
        _dx,
        _dy,
        dt,
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

    if all((i, j) .< size(Vy) .- 1)
        if isvalid_vy(ϕ, i, j + 1)
            θ = 1.0
            # Interpolated Vx into Vy node (includes density gradient)
            Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
            # Vertical velocity
            Vyᵢⱼ = Vy[i + 1, j + 1]
            # Get necessary buoyancy forces
            # i_W, i_E = max(i - 1, 1), min(i + 1, nx)
            j_N = min(j + 1, size(ρgy, 2))
            ρg_S = ρgy[i, j] * ϕ.center[i, j]
            ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
            # Spatial derivatives
            # ∂ρg∂x = (ρg_E - ρg_W) * _dx
            ∂ρg∂y = (ρg_N - ρg_S) * _dy
            # correction term
            # ρg_correction = (Vxᵢⱼ + Vyᵢⱼ * ∂ρg∂y) * θ * dt
            ρg_correction = Vyᵢⱼ * ∂ρg∂y * θ * dt

            Ry[i, j] =
                R_Vy =
                -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
                av_ya(ρgy, ϕ.center) + ρg_correction
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)

            # ρgx_correction = (Vxᵢⱼ) * θ * dt
            # ρgy_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt
            # Ry[i, j] = R_Vy =
            #     -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
            #     av_ya(ρgy, ϕ.center) + ρgx_correction
            # Vy[i + 1, j + 1] += R_Vy * inv(inv(ηdτ / av_ya(ητ)) + ρgy_correction)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end

@parallel_indices (i, j) function compute_V!(
        Vx::AbstractArray{T, 2},
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
        dt,
    ) where {T}
    @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j)
    @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j)
    @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
    @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
    @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)
    @inline harm_xa(A) = _av_xa(A, i, j)
    @inline harm_ya(A) = _av_ya(A, i, j)

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
            θ = 1.0
            # Vertical velocity
            Vyᵢⱼ = Vy[i + 1, j + 1]
            # Get necessary buoyancy forces
            j_N = min(j + 1, size(ρgy, 2))
            ρg_S = ρgy[i, j] * ϕ.center[i, j]
            ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
            # Spatial derivatives
            ∂ρg∂y = (ρg_N - ρg_S) * _dy
            # correction term
            ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt
            Ry[i, j] =
                R_Vy =
                -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
                av_ya(ρgy, ϕ.center) + ρg_correction
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
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
        ϕ::JustRelax.RockRatio,
        _dx,
        _dy,
        _dz,
    ) where {T}
    @inline harm_x(A) = _harm_x(A, i, j, k)
    @inline harm_y(A) = _harm_y(A, i, j, k)
    @inline harm_z(A) = _harm_z(A, i, j, k)
    @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j, k)
    @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j, k)
    @inline d_za(A, ϕ) = _d_za(A, ϕ, _dz, i, j, k)
    @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j, k)
    @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j, k)
    @inline d_zi(A, ϕ) = _d_zi(A, ϕ, _dz, i, j, k)
    @inline av_x(A) = _av_x(A, i, j, k)
    @inline av_y(A) = _av_y(A, i, j, k)
    @inline av_z(A) = _av_z(A, i, j, k)
    @inline av_x(A, ϕ) = _av_x(A, ϕ, i, j, k)
    @inline av_y(A, ϕ) = _av_y(A, ϕ, i, j, k)
    @inline av_z(A, ϕ) = _av_z(A, ϕ, i, j, k)

    @inbounds begin
        if all((i, j, k) .< size(Vx) .- 1)
            if isvalid_vx(ϕ, i + 1, j, k)
                Rx_ijk =
                    Rx[i, j, k] =
                    d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.xy) + d_zi(τxz, ϕ.xz) -
                    d_xa(P, ϕ.center) - av_x(fx, ϕ.center)
                Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
            else
                Rx[i, j, k] = zero(T)
                Vx[i + 1, j + 1, k + 1] = zero(T)
            end
        end
        if all((i, j, k) .< size(Vy) .- 1)
            if isvalid_vy(ϕ, i, j + 1, k)
                Ry_ijk =
                    Ry[i, j, k] =
                    d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.xy) + d_zi(τyz, ϕ.yz) -
                    d_ya(P, ϕ.center) - av_y(fy, ϕ.center)
                Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
            else
                Ry[i, j, k] = zero(T)
                Vy[i + 1, j + 1, k + 1] = zero(T)
            end
        end
        if all((i, j, k) .< size(Vz) .- 1)
            if isvalid_vz(ϕ, i, j, k + 1)
                Rz_ijk =
                    Rz[i, j, k] =
                    d_za(τzz, ϕ.center) + d_xi(τxz, ϕ.xz) + d_yi(τyz, ϕ.yz) -
                    d_za(P, ϕ.center) - av_z(fz, ϕ.center)
                Vz[i + 1, j + 1, k + 1] += Rz_ijk * ηdτ / av_z(ητ)
            else
                Rz[i, j, k] = zero(T)
                Vz[i + 1, j + 1, k + 1] = zero(T)
            end
        end
    end

    return nothing
end
