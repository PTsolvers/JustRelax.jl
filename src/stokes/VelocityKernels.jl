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
    Vx::AbstractArray{T,2}, Vy, Vx_on_Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy, dt
) where {T}
    d_xi(A) = _d_xi(A, i, j, _dx)
    d_yi(A) = _d_yi(A, i, j, _dy)
    d_xa(A) = _d_xa(A, i, j, _dx)
    d_ya(A) = _d_ya(A, i, j, _dy)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)

    if all((i, j) .< size(Vx) .- 1)
        Vx[i + 1, j + 1] +=
            (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
    end

    if all((i, j) .< size(Vy) .- 1)
        θ = 1.0
        # Interpolate Vx into Vy node
        Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        i_W, i_E = max(i - 1, 1), min(i + 1, nx)
        j_N = min(j + 1, ny)
        ρg_stencil = (
            ρgy[i_W, j], ρgy[i, j], ρgy[i_E, j], ρgy[i_W, j_N], ρgy[i, j_N], ρgy[i_E, j_N]
        )
        ρg_W = (ρg_stencil[1] + ρg_stencil[2] + ρg_stencil[4] + ρg_stencil[5]) * 0.25
        ρg_E = (ρg_stencil[2] + ρg_stencil[3] + ρg_stencil[5] + ρg_stencil[6]) * 0.25
        ρg_S = ρg_stencil[2]
        ρg_N = ρg_stencil[5]
        # Spatial derivatives
        ∂ρg∂x = (ρg_E - ρg_W) * _dx
        ∂ρg∂y = (ρg_N - ρg_S) * _dy
        # correction term
        ρg_correction = (Vxᵢⱼ * ∂ρg∂x + Vyᵢⱼ * ∂ρg∂y) * θ * dt

        Vy[i + 1, j + 1] +=
            (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy) + ρg_correction) * ηdτ /
            av_ya(ητ)
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
    Rx::AbstractArray{T,2}, Ry, Vx, Vy, Vx_on_Vy, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, dt
) where {T}
    @inline d_xa(A) = _d_xa(A, i, j, _dx)
    @inline d_ya(A) = _d_ya(A, i, j, _dy)
    @inline d_xi(A) = _d_xi(A, i, j, _dx)
    @inline d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        end
        if all((i, j) .≤ size(Ry))
            θ = 1.0
            #     0.25 * (Vx[i, j + 1] + Vx[i + 1, j + 1] + Vx[i, j + 2] + Vx[i + 1, j + 2])
            Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
            # Vertical velocity
            Vyᵢⱼ = Vy[i + 1, j + 1]
            # Get necessary buoyancy forces
            i_W, i_E = max(i - 1, 1), min(i + 1, nx)
            j_N = min(j + 1, ny)
            ρg_stencil = (
                ρgy[i_W, j],
                ρgy[i, j],
                ρgy[i_E, j],
                ρgy[i_W, j_N],
                ρgy[i, j_N],
                ρgy[i_E, j_N],
            )
            ρg_W = (ρg_stencil[1] + ρg_stencil[2] + ρg_stencil[4] + ρg_stencil[5]) * 0.25
            ρg_E = (ρg_stencil[2] + ρg_stencil[3] + ρg_stencil[5] + ρg_stencil[6]) * 0.25
            ρg_S = ρg_stencil[2]
            ρg_N = ρg_stencil[5]
            # Spatial derivatives
            ∂ρg∂x = (ρg_E - ρg_W) * _dx
            ∂ρg∂y = (ρg_N - ρg_S) * _dy
            # correction term
            ρg_correction = (Vxᵢⱼ * ∂ρg∂x + Vyᵢⱼ * ∂ρg∂y) * θ * dt
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy) + ρg_correction
            # Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        end
    end

    return nothing
end

@parallel_indices (i) function FreeSurface_Vy!(
    Vy::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    P::AbstractArray{T,2},
    η::AbstractArray{T,2},
    dx,
    dy,
) where {T}
    Vy[i, end] = Vy[i, end-1] + 3.0/2.0*(P[i, end]/(2.0*η[i, end]) + inv(3.0) * (Vx[i+1, end]-Vx[i, end])*inv(dx))*dy
    return nothing
end

@parallel_indices (i) function FreeSurface_Vy_ve!(
    Vy::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    P::AbstractArray{T,2},
    P_old::AbstractArray{T,2},
    τyy_old::AbstractArray{T,2},
    η::AbstractArray{T,2},
    rheology,
    phase_ratios::CellArray,
    dt::T,
    dx::T,
    dy::T
) where {T}
    phase = @inbounds phase_ratios[i,end]
    Gdt = (fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Vy[i, end] = Vy[i, end-1] + 3.0/2.0*(P[i, end]/(2.0*η[i, end]) - (τyy_old[i, end]+P_old[i, end])/(2.0*Gdt) + inv(3.0) * (Vx[i+1, end]-Vx[i, end])*inv(dx))*dy
    return nothing
end

@parallel_indices (i) function FreeSurface_Vy_vep!(
    Vy::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    P::AbstractArray{T,2},
    P_old::AbstractArray{T,2},
    τyy_old::AbstractArray{T,2},
    εII_pl::AbstractArray{T,2},
    η::AbstractArray{T,2},
    rheology,
    phase_ratios::CellArray,
    dt::T,
    dx::T,
    dy::T
) where {T}
    phase = @inbounds phase_ratios[i, end]
    Gdt = (fn_ratio(get_shear_modulus, rheology, phase) * dt)
    Vy[i, end] = Vy[i, end-1] + 3.0/2.0*(P[i, end]/(2.0*η[i, end]*(1.0-εII_pl[i, end])) - (τyy_old[i, end]+P_old[i, end])/(2.0*Gdt) + inv(3.0) * (Vx[i+1, end]-Vx[i, end])*inv(dx))*dy
    return nothing
end

@parallel_indices (i) function FreeSurface_τyy!(σyy::AbstractArray{T,2},) where {T}
    σyy[i, end] = 0.0
    return
end
@parallel_indices (i) function FreeSurface_τxy!(σxy::AbstractArray{T,2},) where {T}
    σxy[i, end] =  -σxy[i, end-1]
    return
end
