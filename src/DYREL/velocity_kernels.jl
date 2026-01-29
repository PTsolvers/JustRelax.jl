## RESIDUALS

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2}, Ry, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
    ) where {T}
    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

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
        Rx::AbstractArray{T, 2}, Ry, Vx, Vy, P, ΔPψ, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy, dt
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

    Base.@propagate_inbounds @inline d_xa(A) = _d_xa(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_ya(A) = _d_ya(A, _dy, i, j)
    Base.@propagate_inbounds @inline d_xi(A) = _d_xi(A, _dx, i, j)
    Base.@propagate_inbounds @inline d_yi(A) = _d_yi(A, _dy, i, j)
    Base.@propagate_inbounds @inline av_xa(A) = _av_xa(A, i, j)
    Base.@propagate_inbounds @inline av_ya(A) = _av_ya(A, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            Rx[i, j] = (d_xa(τxx) + d_yi(τxy) - d_xa(P) - d_xa(P_num) - d_xa(ΔPψ) - av_xa(ρgx)) / Dx[i, j]
        end
        if all((i, j) .≤ size(Ry))
            Ry[i, j] = (d_ya(τyy) + d_xi(τxy) - d_ya(P) - d_ya(P_num) - d_ya(ΔPψ) - av_ya(ρgy)) / Dy[i, j]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_V_damping!(
        dVdτ::NTuple{N, AbstractArray{T, N}}, R::NTuple{N, AbstractArray{T, N}}, αV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}

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
    ) where {N, T}

    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(dVdτ[i]))
            V[i][I .+ 1...] += dVdτ[i][I...] * βV[i][I...] * dτV[i][I...]
        end
    end

    return nothing
end
