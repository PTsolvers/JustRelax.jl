## VELOCITY

# @parallel_indices (i, j) function compute_V!(
#         Vx::AbstractArray{T, 2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy
#     ) where {T}
#     Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
#     Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
#     Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
#     Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
#     Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
#     Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
#     Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
#     Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

#     if all((i, j) .< size(Vx) .- 1)
#         @inbounds Vx[i + 1, j + 1] +=
#             (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
#     end
#     if all((i, j) .< size(Vy) .- 1)
#         @inbounds Vy[i + 1, j + 1] +=
#             (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy)) * ηdτ / av_ya(ητ)
#     end
#     return nothing
# end

# # with free surface stabilization
# @parallel_indices (i, j) function compute_V!(
#         Vx::AbstractArray{T, 2}, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy, dt
#     ) where {T}
#     Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
#     Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
#     Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
#     Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
#     Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
#     Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)
#     Base.@propagate_inbounds @inline harm_xa(A) = _av_xa(A, i, j)
#     Base.@propagate_inbounds @inline harm_ya(A) = _av_ya(A, i, j)

#     nx, ny = size(ρgy)

#     if all((i, j) .< size(Vx) .- 1)
#         @inbounds Vx[i + 1, j + 1] +=
#             (-d_xa(P) + d_xa(τxx) + d_yi(τxy) - av_xa(ρgx)) * ηdτ / av_xa(ητ)
#     end

#     @inbounds if all((i, j) .< size(Vy) .- 1)
#         θ = 1.0
#         # Interpolated Vx into Vy node (includes density gradient)
#         # Vertical velocity
#         Vyᵢⱼ = Vy[i + 1, j + 1]
#         # Get necessary buoyancy forces
#         j_N = min(j + 1, ny)
#         ρg_S = ρgy[i, j]
#         ρg_N = ρgy[i, j_N]
#         # Spatial derivatives
#         ∂ρg∂y = (ρg_N - ρg_S) * _dy
#         # correction term
#         ρg_correction = Vyᵢⱼ * ∂ρg∂y * θ * dt

#         Vy[i + 1, j + 1] +=
#             (-d_ya(P) + d_ya(τyy) + d_xi(τxy) - av_ya(ρgy) + ρg_correction) * ηdτ /
#             av_ya(ητ)
#     end

#     return nothing
# end

# @parallel_indices (i, j, k) function compute_V!(
#         Vx::AbstractArray{T, 3},
#         Vy,
#         Vz,
#         Rx,
#         Ry,
#         Rz,
#         P,
#         fx,
#         fy,
#         fz,
#         τxx,
#         τyy,
#         τzz,
#         τyz,
#         τxz,
#         τxy,
#         ητ,
#         ηdτ,
#         _dx,
#         _dy,
#         _dz,
#     ) where {T}
#     Base.@propagate_inbounds @inline harm_x(A) = _harm_x(A, i, j, k)
#     Base.@propagate_inbounds @inline harm_y(A) = _harm_y(A, i, j, k)
#     Base.@propagate_inbounds @inline harm_z(A) = _harm_z(A, i, j, k)
#     Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
#     Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
#     Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)
#     Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j, k)
#     Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j, k)
#     Base.@propagate_inbounds @inline d_za(A) = _d_za(A, _dz, i, j, k)

#     @inbounds begin
#         if all((i, j, k) .≤ size(Rx))
#             Rx_ijk =
#                 Rx[i, j, k] =
#                 d_xa(τxx) +
#                 _dy * (τxy[i + 1, j + 1, k] - τxy[i + 1, j, k]) +
#                 _dz * (τxz[i + 1, j, k + 1] - τxz[i + 1, j, k]) - d_xa(P) - av_x(fx)
#             Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
#         end
#         if all((i, j, k) .≤ size(Ry))
#             Ry_ijk =
#                 Ry[i, j, k] =
#                 _dx * (τxy[i + 1, j + 1, k] - τxy[i, j + 1, k]) +
#                 _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
#                 _dz * (τyz[i, j + 1, k + 1] - τyz[i, j + 1, k]) - d_ya(P) - av_y(fy)
#             Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
#         end
#         if all((i, j, k) .≤ size(Rz))
#             Rz_ijk =
#                 Rz[i, j, k] =
#                 _dx * (τxz[i + 1, j, k + 1] - τxz[i, j, k + 1]) +
#                 _dy * (τyz[i, j + 1, k + 1] - τyz[i, j, k + 1]) +
#                 d_za(τzz) - d_za(P) - av_z(fz)
#             Vz[i + 1, j + 1, k + 1] += Rz_ijk * ηdτ / av_z(ητ)
#         end
#     end

#     return nothing
# end

## RESIDUALS

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A)  = JustRelax2D._d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A)  = JustRelax2D._d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A)  = JustRelax2D._d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A)  = JustRelax2D._d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = JustRelax2D._av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = JustRelax2D._av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
        end
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, Vx, Vy, P, ΔPψ,τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, dt
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    nx, ny = size(ρgy)
    if all((i, j) .≤ size(Rx))
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(ΔPψ) - av_xa(ρgx)
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

        Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(ΔPψ) - av_ya(ρgy) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, P_num, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, Dx, Dy, _dx, _dy
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A)  = JustRelax2D._d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A)  = JustRelax2D._d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A)  = JustRelax2D._d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A)  = JustRelax2D._d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = JustRelax2D._av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = JustRelax2D._av_ya(A, i, j)

    # @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = (d_xa(τxx) + d_yi(τxy) - d_xa(P)- d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j]
        end
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = (d_ya(τyy) + d_xi(τxy) - d_ya(P)- d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j]
        end
    # end

    return nothing
end

@parallel_indices (I...) function update_V_damping!(
        dVdτ::NTuple{N, AbstractArray{T, N}}, R::NTuple{N, AbstractArray{T, N}}, αV::NTuple{N, AbstractArray{T, N}}
    ) where {N,T}

    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(R[i]))
            dVdτ[i][I...] = αV[i][I...] * dVdτ[i][I...] + R[i][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_DR_V!(
        V::NTuple{N, AbstractArray{T, N}}, dVdτ::NTuple{N, AbstractArray{T, N}}, βV::NTuple{N, AbstractArray{T, N}}, dτV::NTuple{N, AbstractArray{T, N}}
    ) where {N,T}

    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(dVdτ[i]))
            V[i][I .+ 1...] += dVdτ[i][I...] * βV[i][I...] * dτV[i][I...] 
        end
    end

    return nothing
end
