## DIVERGENCE + DEVIATORIC STRAIN RATE TENSOR

function compute_∇V_strain_rate!(stokes, _di, ni, dim)
    @parallel (@idx ni .+ 1) compute_∇V_strain_rate!(
        stokes.∇V,
        @strain(stokes)...,
        @velocity(stokes)...,
        _di.vertex,
        _di.velocity...
    )
    return interpolate_shear_ε_to_centers(stokes, dim)
end

function interpolate_shear_ε_to_centers(stokes, ::Val{2})
    vertex2center!(stokes.ε.xy_c, stokes.ε.xy)
    return nothing
end

function interpolate_shear_ε_to_centers(stokes, ::Val{3})
    vertex2center!(stokes.ε.yz_c, stokes.ε.yz)
    vertex2center!(stokes.ε.xz_c, stokes.ε.xz)
    vertex2center!(stokes.ε.xy_c, stokes.ε.xy)
    return nothing
end

@parallel_indices (i, j) function compute_∇V_strain_rate!(
        ∇V::AbstractArray{T, 2},
        εxx::AbstractArray{T, 2},
        εyy,
        εxy,
        Vx,
        Vy,
        _di_vertex,
        _di_vx,
        _di_vy,
    ) where {T}

    third = T(1) / T(3)

    @inbounds begin
        vx_s = Vx[i, j]
        vx_n = Vx[i, j + 1]
        vy_w = Vy[i, j]
        vy_e = Vy[i + 1, j]

        if i ≤ size(εxy, 1) && j ≤ size(εxy, 2)
            _dy_vx = @dy(_di_vx, j)
            _dx_vy = @dx(_di_vy, i)

            dVx_dy = (vx_n - vx_s) * _dy_vx
            dVy_dx = (vy_e - vy_w) * _dx_vy
            εxy[i, j] = 0.5 * (dVx_dy + dVy_dx)
        end

        if i ≤ size(∇V, 1) && j ≤ size(∇V, 2)
            vx_ne = Vx[i + 1, j + 1]
            vy_ne = Vy[i + 1, j + 1]
            _dx, _dy = @dxi(_di_vertex, i, j)

            dVx_dx = (vx_ne - vx_n) * _dx
            dVy_dy = (vy_ne - vy_e) * _dy
            div_ij = dVx_dx + dVy_dy
            ∇V[i, j] = div_ij

            div_third = div_ij * third
            εxx[i, j] = dVx_dx - div_third
            εyy[i, j] = dVy_dy - div_third
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_∇V_strain_rate!(
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
        _di_vertex,
        _di_vx,
        _di_vy,
        _di_vz,
    ) where {T}

    third = T(1) / T(3)

    @inbounds begin
        if all((i, j, k) .≤ size(∇V))
            _dx, _dy, _dz = @dxi(_di_vertex, i, j, k)
            dVx_dx = (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) * _dx
            dVy_dy = (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) * _dy
            dVz_dz = (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) * _dz
            div_ijk = dVx_dx + dVy_dy + dVz_dz
            ∇V[i, j, k] = div_ijk

            div_third = div_ijk * third
            εxx[i, j, k] = dVx_dx - div_third
            εyy[i, j, k] = dVy_dy - div_third
            εzz[i, j, k] = dVz_dz - div_third
        end

        if all((i, j, k) .≤ size(εyz))
            _dz_vy = @dz(_di_vy, k)
            _dy_vz = @dy(_di_vz, j)
            εyz[i, j, k] =
                0.5 * (
                _dz_vy * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy_vz * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
            )
        end

        if all((i, j, k) .≤ size(εxz))
            _dz_vx = @dz(_di_vx, k)
            _dx_vz = @dx(_di_vz, i)
            εxz[i, j, k] =
                0.5 * (
                _dz_vx * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx_vz * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
            )
        end

        if all((i, j, k) .≤ size(εxy))
            _dy_vx = @dy(_di_vx, j)
            _dx_vy = @dx(_di_vy, i)
            εxy[i, j, k] =
                0.5 * (
                _dy_vx * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx_vy * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
            )
        end
    end

    return nothing
end

## RESIDUALS

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _di_center, _di_vertex
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        _dx_c = @dx(_di_center, i)
        _dy_v = @dy(_di_vertex, j)
        Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx_c, i, j)
        Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy_v, i, j)
        Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        _dy_c = @dy(_di_center, j)
        _dx_v = @dx(_di_vertex, i)
        Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy_c, i, j)
        Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx_v, i, j)
        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy)
    end
    # end
    return nothing
end

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        Vx,
        Vy,
        P,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        _di_center,
        _di_vertex,
        dt,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        _dx_c = @dx(_di_center, i)
        _dy_v = @dy(_di_vertex, j)
        Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx_c, i, j)
        Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy_v, i, j)
        Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
    end

    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        _dy_c = @dy(_di_center, j)
        _dx_v = @dx(_di_vertex, i)
        Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy_c, i, j)
        Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx_v, i, j)
        θ = 1.0
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        j_N = min(j + 1, ny)
        ρg_S = ρgy[i, j]
        ρg_N = ρgy[i, j_N]
        # Spatial derivatives
        ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
        # correction term
        ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt

        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        P,
        P_num,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        Dx,
        Dy,
        _di_center,
        _di_vertex,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        _dx_c = @dx(_di_center, i)
        _dy_v = @dy(_di_vertex, j)
        Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx_c, i, j)
        Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy_v, i, j)
        Rx[i, j] = (d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j]
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        _dy_c = @dy(_di_center, j)
        _dx_v = @dx(_di_vertex, i)
        Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy_c, i, j)
        Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx_v, i, j)
        Ry[i, j] = (d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j]
    end
    # end

    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _di
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        _dx = @dx(_di_center, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dy(_di_vertex, k)

        Rx[i, j, k] = d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) - d_xa(P, _dx) - d_xa(ΔPψ, _dx) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dy(_di_vertex, k)

        Ry[i, j, k] = d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) - d_ya(P, _dy) - d_ya(ΔPψ, _dy) - av_ya(ρgy)
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        _dx_v = @dx(_di_vertex, i)
        _dy_v = @dy(_di_center, j)
        _dz_c = @dy(_di_center, k)


        Rz[i, j, k] = d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) - d_za(P, _dz) - d_za(ΔPψ, _dz) - av_za(ρgz)
    end
    # end
    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, Vx, Vy, Vz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _di, dt
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j)

    nx, ny, nz = size(ρgz)
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        _dx = @dx(_di_center, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dy(_di_vertex, k)

        Rx[i, j, k] = d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) - d_xa(P, _dx) - d_xa(ΔPψ, _dx) - av_xa(ρgx)
    end

    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dy(_di_vertex, k)

        Ry[i, j, k] = d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) - d_ya(P, _dy) - d_ya(ΔPψ, _dy) - av_ya(ρgy)
    end

    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dy(_di_center, k)

        θ = 1.0
        # Vertical velocity
        Vzᵢⱼₖ = Vz[i + 1, j + 1, k + 1]
        # Get necessary buoyancy forces
        k_T = min(k + 1, nz)
        ρg_B = ρgz[i, j, k]
        ρg_T = ρgz[i, j, k_T]
        # Spatial derivatives
        ∂ρg∂z = (ρg_T - ρg_B) * _dz
        # correction term
        ρg_correction = (Vzᵢⱼₖ * ∂ρg∂z) * θ * dt

        Rz[i, j, k] = d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) - d_za(P, _dz) - d_za(ΔPψ, _dz) - av_za(ρgz, _dz) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, P, P_num, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, Dx, Dy, Dz, _di
    ) where {T}
    _dx, _dy, _dz = @dxi(_di, i, j, k)

    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j, k)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j, k)
    Base.@propagate_inbounds @inline av_za(A) = _av_za(A, i, j, k)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        Rx[i, j, k] = (d_xa(τxx) + d_yi(τxy) + d_zi(τxz) - d_xa(P) - d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j, k]
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        Ry[i, j, k] = (d_ya(τyy) + d_xi(τxy) + d_zi(τyz) - d_ya(P) - d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j, k]
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        Rz[i, j, k] = (d_za(τzz) + d_xi(τxz) + d_yi(τyz) - d_za(P) - d_za(P_num) - d_za(ΔPψ) - av_za(ρgz)) / Dz[i, j, k]
    end
    # end

    return nothing
end

@parallel_indices (I...) function update_V_damping_DR_V!(
        V::NTuple{N, AbstractArray{T, N}},
        dVdτ::NTuple{N, AbstractArray{T, N}},
        R::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(R[d]))
            dVdτ[d][I...] = αV[d][I...] * dVdτ[d][I...] + R[d][I...]
            V[d][I .+ 1...] += dVdτ[d][I...] * βV[d][I...] * dτV[d][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function compute_dV!(
        dV::NTuple{N, AbstractArray{T, N}},
        dVdτ::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(dV[d]))
            dV[d][I...] = dVdτ[d][I...] * βV[d][I...] * dτV[d][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_cV!(
        cV::NTuple{N, AbstractArray{T, N}}, cV_I
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(cV[d]))
            cV[d][I...] = cV_I
        end
    end

    return nothing
end
