@parallel_indices (i, j) function compute_strain_rateAD!(
        εxx::AbstractArray{T, 2}, εyy, εxy, ∇V, Vx, Vy, ϕ::JustRelax.RockRatio, _dx, _dy
    ) where {T}
    @inline d_xi(A) = _d_xi(A, _dx, i, j)
    @inline d_yi(A) = _d_yi(A, _dy, i, j)
    @inline d_xa(A) = _d_xa(A, _dx, i, j)
    @inline d_ya(A) = _d_ya(A, _dy, i, j)

    if all((i, j) .≤ size(εxx))
        if isvalid_c(ϕ, i, j)
            ∇V_ij = (d_xi(Vx) + d_yi(Vy)) / 3.0
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

@parallel_indices (i, j) function compute_Res!(
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
            #Vx[i + 1, j + 1] += R_Vx * ηdτ / av_xa(ητ)
        else
            Rx[i, j] = zero(T)
            #Vx[i + 1, j + 1] = zero(T)
        end
    end

    if all((i, j) .< size(Vy) .- 1)
        if isvalid_vy(ϕ, i, j + 1)
        #=    θ = 1.0
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
        =#    Ry[i, j] =
                R_Vy =
                -d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) -
                av_ya(ρgy, ϕ.center)# + ρg_correction
            #Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ)
        else
            Ry[i, j] = zero(T)
            #Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end

@parallel_indices (i, j) function update_V!(
        Vx::AbstractArray{T, 2},
        Vy,
        Rx,
        Ry,
        ηdτ,
        ητ,
        ϕ::JustRelax.RockRatio,
    ) where {T}

    @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)
    @inline av_xa(A) = _av_xa(A, i, j)
    @inline av_ya(A) = _av_ya(A, i, j)
    @inline harm_xa(A) = _av_xa(A, i, j)
    @inline harm_ya(A) = _av_ya(A, i, j)

    if all((i, j) .< size(Vx) .- 1)
        if isvalid_vx(ϕ, i + 1, j)
            Vx[i + 1, j + 1] += Rx[i + 1, j + 1] * ηdτ / av_xa(ητ)
        else
            Rx[i, j] = zero(T)
            Vx[i + 1, j + 1] = zero(T)
        end
    end

    if all((i, j) .< size(Vy) .- 1)
        if isvalid_vy(ϕ, i, j + 1)
            Vy[i + 1, j + 1] += Ry[i + 1, j + 1] * ηdτ / av_ya(ητ)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end
