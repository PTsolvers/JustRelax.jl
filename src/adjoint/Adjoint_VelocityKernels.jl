# with free surface stabilization

@parallel_indices (i, j) function update_V!(
        Vx::AbstractArray{T, 2}, Vy, ResVx, ResVy, ηdτ, ρgx, ρgy, ητ, _dx, _dy
    ) where {T}
    d_xi(A) = _d_xi(A, _dx, i, j)
    d_yi(A) = _d_yi(A, _dy, i, j)
    d_xa(A) = _d_xa(A, _dx, i, j)
    d_ya(A) = _d_ya(A, _dy, i, j)
    av_xa(A) = _av_xa(A, i, j)
    av_ya(A) = _av_ya(A, i, j)
    harm_xa(A) = _av_xa(A, i, j)
    harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        Vx[i + 1, j + 1] += ResVx[i+1,j+1] * ηdτ / av_xa(ητ)
    end
    if all((i, j) .< size(Vy) .- 1)
        Vy[i + 1, j + 1] += ResVy[i+1,j+1] * ηdτ / av_ya(ητ)
    end
    return nothing
end
#=
@parallel_indices (i, j) function update_V!(
    Vx::AbstractArray{T,2}, Vy, ResVx, ResVy, Vx_on_Vy, ηdτ, ρgy, ητ, _dx, _dy, dt
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
        Vx[i + 1, j + 1] += ResVx[i + 1, j + 1] * ηdτ / av_xa(ητ)
    end

    if all((i, j) .< size(Vy) .- 1)

        θ = 1.0
        # Interpolated Vx into Vy node (includes density gradient)
        Vxᵢⱼ = Vx_on_Vy[i + 1, j + 1]
        # Vertical velocity
        Vyᵢⱼ = Vy[i + 1, j + 1]
        # Get necessary buoyancy forces
        # i_W, i_E = max(i - 1, 1), min(i + 1, nx)
        j_N = min(j + 1, ny)
        # ρg_stencil = (
        #     ρgy[i_W, j], ρgy[i, j], ρgy[i_E, j], ρgy[i_W, j_N], ρgy[i, j_N], ρgy[i_E, j_N]
        # )
        # ρg_W = (ρg_stencil[1] + ρg_stencil[2] + ρg_stencil[4] + ρg_stencil[5]) * 0.25
        # ρg_E = (ρg_stencil[2] + ρg_stencil[3] + ρg_stencil[5] + ρg_stencil[6]) * 0.25
        ρg_S = ρgy[i, j]
        ρg_N = ρgy[i, j_N]
        # Spatial derivatives
        # ∂ρg∂x = (ρg_E - ρg_W) * _dx
        ∂ρg∂y = (ρg_N - ρg_S) * _dy
        # correction term
        ρg_correction = (Vxᵢⱼ + Vyᵢⱼ * ∂ρg∂y) * θ * dt

        Vy[i + 1, j + 1] += (ResVy[i + 1, j + 1] + ρg_correction) * ηdτ / av_ya(ητ)
    end

    return nothing
end
=#

@parallel_indices (i, j) function compute_strain_rateAD!(
    εxx::AbstractArray{T,2}, εyy, εxy, Vx, Vy, _dx, _dy
) where {T}
    d_xi(A) = _d_xi(A, _dx, i, j)
    d_yi(A) = _d_yi(A, _dy, i, j)
    d_xa(A) = _d_xa(A, _dx, i, j)
    d_ya(A) = _d_ya(A, _dy, i, j)

    if all((i, j) .≤ size(εxx))
        ∇V_ij = (d_xi(Vx) + d_yi(Vy)) / 3.0
        εxx[i, j] = d_xi(Vx) - ∇V_ij
        εyy[i, j] = d_yi(Vy) - ∇V_ij
    end
    εxy[i, j] = 0.5 * (d_ya(Vx) + d_xa(Vy))

    return nothing
end
