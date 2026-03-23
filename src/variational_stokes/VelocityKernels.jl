"""
    compute_∇V!(∇V, V, ϕ, _di)

Compute the divergence of the velocity field `V` and store it in `∇V`, taking into account the rock ratio `ϕ` and grid spacing `_di`.
"""
@parallel_indices (I...) function compute_∇V!(
        ∇V::AbstractArray{T, N}, V::NTuple{N}, ϕ::JustRelax.RockRatio, _di::NTuple{N}
    ) where {T, N}
    @inbounds ∇V[I...] =
        isvalid_c(ϕ, I...) ? div(V..., @dxi(_di, I...)..., I...) : zero(T)
    return nothing
end

"""
    compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, ϕ, _dx, _dy)

Compute the components of the strain rate tensor `ε` from the velocity field `V` and its divergence `∇V`, taking into account the rock ratio `ϕ` and grid spacing `_dx`, `_dy`.
"""
@parallel_indices (i, j) function compute_strain_rate!(
        εxx::AbstractArray{T, 2},
        εyy,
        εxy,
        ∇V,
        Vx,
        Vy,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vx,
        _di_vy,
    ) where {T}
    _dx, _dy = @dxi(_di_center, i, j)
    _dy_vx = @dy(_di_vx, j)
    _dx_vy = @dx(_di_vy, i)

    Vx1 = Vx[i, j]
    Vx2 = Vx[i, j + 1]
    Vy1 = Vy[i, j]
    Vy2 = Vy[i + 1, j]
    if all((i, j) .≤ size(εxx))
        @inbounds if isvalid_c(ϕ, i, j)
            Vx3 = Vx[i + 1, j + 1]
            Vy3 = Vy[i + 1, j + 1]

            ∇V_ij = ∇V[i, j] / 3
            εxx[i, j] = (Vx3 - Vx2) * _dx - ∇V_ij
            εyy[i, j] = (Vy3 - Vy2) * _dy - ∇V_ij
        else
            εxx[i, j] = zero(T)
            εyy[i, j] = zero(T)
        end
    end
    @inbounds if isvalid_v(ϕ, i, j)
        εxy[i, j] = 0.5 * ((Vx2 - Vx1) * _dy_vx + (Vy2 - Vy1) * _dx_vy)
    else
        εxy[i, j] = zero(T)
    end

    return nothing
end

"""
    compute_strain_rate_from_increment!(εxx, εyy, εxy, Δεxx, Δεyy, Δεxy, ϕ, _dt)

Compute the components of the strain rate tensor `ε` from the strain increments `Δε`, taking into account the rock ratio `ϕ` and time step `_dt`.
"""
@parallel_indices (i, j) function compute_strain_rate_from_increment!(
        εxx::AbstractArray{T, 2}, εyy, εxy, Δεxx, Δεyy, Δεxy, ϕ::JustRelax.RockRatio, _dt
    ) where {T}

    if all((i, j) .≤ size(εxx))
        if isvalid_c(ϕ, i, j)
            εxx[i, j] = Δεxx[i, j] * _dt
            εyy[i, j] = Δεyy[i, j] * _dt
        else
            εxx[i, j] = zero(T)
            εyy[i, j] = zero(T)
        end

    end

    εxy[i, j] = if isvalid_v(ϕ, i, j)
        Δεxy[i, j] * _dt
    else
        zero(T)
    end


    return nothing
end

"""
    compute_strain_rate!(εxx, εyy, εzz, εyz, εxz, εxy, ∇V, Vx, Vy, Vz, ϕ, _dx, _dy, _dz)

Compute the 3D components of the strain rate tensor `ε` from the velocity field `V` and its divergence `∇V`, taking into account the rock ratio `ϕ` and grid spacing `_dx`, `_dy`, `_dz`.
"""
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
        _di,
    ) where {T}
    _dx, _dy, _dz = @dxi(_di, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)

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

"""
    compute_V!(Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, ϕ, _dx, _dy)

Compute the velocity field `V` from the pressure `P`, stress components `τ`, and other parameters, taking into account the rock ratio `ϕ` and grid spacing `_dx`, `_dy`.
"""
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
        _di_center,
        _di_vertex,
    ) where {T}
    _dx_c, _dy_c = @dxi(_di_center, i, j)
    _dx_v, _dy_v = @dxi(_di_vertex, i, j)
    Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
    Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
    Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
    Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

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

"""
    compute_Vx!(Vx, Rx, P, τxx, τxy, ηdτ, ρgx, ητ, ϕ, _dx, _dy)

Compute the x-component of the velocity field `Vx` from the pressure `P`, stress components `τ`, and other parameters, taking into account the rock ratio `ϕ` and grid spacing `_dx`, `_dy`.
"""
@parallel_indices (i, j) function compute_Vx!(
        Vx::AbstractArray{T, 2},
        Rx,
        P,
        τxx,
        τxy,
        ηdτ,
        ρgx,
        ητ,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vertex,
    ) where {T}
    _dx_c, _dy_c = @dxi(_di_center, i, j)
    _dx_v, _dy_v = @dxi(_di_vertex, i, j)
    Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
    Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
    Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
    Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

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

"""
    compute_Vy!(Vy, Vx_on_Vy, Ry, P, τyy, τxy, ηdτ, ρgy, ητ, ϕ, _dx, _dy, dt)

Compute the y-component of the velocity field `Vy` from the pressure `P`, stress components `τ`, and other parameters, taking into account the rock ratio `ϕ`, grid spacing `_dx`, `_dy`, and time step `dt`.
"""
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
        _di_center,
        _di_vertex,
        dt,
    ) where {T}
    _dx_c, _dy_c = @dxi(_di_center, i, j)
    _dx_v, _dy_v = @dxi(_di_vertex, i, j)
    Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
    Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
    Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
    Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .< size(Vy) .- 1)
            if isvalid_vy(ϕ, i, j + 1)
                θ = 1.0
                # Interpolated Vx into Vy node (includes density gradient)
                Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
                # Vertical velocity
                Vyᵢⱼ = Vy[i + 1, j + 1]
                # Get necessary buoyancy forces
                j_N = min(j + 1, size(ρgy, 2))
                ρg_S = ρgy[i, j] * ϕ.center[i, j]
                ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
                # Spatial derivatives
                ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
                # correction term
                # ρg_correction = (Vxᵢⱼ + Vyᵢⱼ * ∂ρg∂y) * θ * dt
                ρg_correction = Vyᵢⱼ * ∂ρg∂y * θ * dt

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
    end
    return nothing
end

"""
    compute_V!(Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, ϕ, _dx, _dy, dt)

Compute the velocity field `V` with the timestep dt from the pressure `P`, stress components `τ`, and other parameters, taking into account the rock ratio `ϕ`, grid spacing `_dx`, `_dy`, and time step `dt`.
"""
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
        _di_center,
        _di_vertex,
        dt,
    ) where {T}
    _dx_c, _dy_c = @dxi(_di_center, i, j)
    _dx_v, _dy_v = @dxi(_di_vertex, i, j)
    Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
    Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
    Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
    Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
    Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        @inbounds if isvalid_vx(ϕ, i + 1, j)
            Rx[i, j] =
                R_Vx = @inbounds (
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
        @inbounds if isvalid_vy(ϕ, i, j + 1)
            θ = 1.0
            # Vertical velocity
            Vyᵢⱼ = Vy[i + 1, j + 1]
            # Get necessary buoyancy forces
            j_N = min(j + 1, size(ρgy, 2))
            ρg_S = ρgy[i, j] * ϕ.center[i, j]
            ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
            # Spatial derivatives
            ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
            # correction term
            ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt
            Ry[i, j] =
                R_Vy =
                @inbounds -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
                av_ya(ρgy, ϕ.center) + ρg_correction
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end
    return nothing
end

"""
    compute_V!(Vx, Vy, Vz, Rx, Ry, Rz, P, fx, fy, fz, τxx, τyy, τzz, τyz, τxz, τxy, ητ, ηdτ, ϕ, _dx, _dy, _dz)

Compute the 3D velocity field `V` from the pressure `P`, stress components `τ`, body forces `f`, and other parameters, with the rock ratio `ϕ` and grid spacing `_dx`, `_dy`, `_dz`.
"""
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
        _di,
    ) where {T}
    _dx, _dy, _dz = @dxi(_di, i, j, k)
    Base.@propagate_inbounds @inline harm_x(A) = _harm_x(A, i, j, k)
    Base.@propagate_inbounds @inline harm_y(A) = _harm_y(A, i, j, k)
    Base.@propagate_inbounds @inline harm_z(A) = _harm_z(A, i, j, k)
    Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A, ϕ) = _d_za(A, ϕ, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A, ϕ) = _d_zi(A, ϕ, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)
    Base.@propagate_inbounds @inline av_x(A, ϕ) = _av_x(A, ϕ, i, j, k)
    Base.@propagate_inbounds @inline av_y(A, ϕ) = _av_y(A, ϕ, i, j, k)
    Base.@propagate_inbounds @inline av_z(A, ϕ) = _av_z(A, ϕ, i, j, k)

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
