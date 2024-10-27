# @parallel_indices (I...) function compute_∇V!(
#     ∇V::AbstractArray{T,2}, Vx, Vy, ϕ::JustRelax.RockRatio, _dx, _dy
# ) where {T}
#     @inline d_xi(A) = _d_xi(A, _dx, I...)
#     @inline d_yi(A) = _d_yi(A, _dy, I...)

#     if isvalid_c(ϕ, I...)
#         @inbounds ∇V[I...] = d_xi(Vx) + d_yi(Vy)
#     else
#         @inbounds ∇V[I...] = zero(T)
#     end
#     return nothing
# end

# @parallel_indices (I...) function compute_∇V!(
#     ∇V::AbstractArray{T,2}, Vx, Vy, Vz, ϕ::JustRelax.RockRatio, _dx, _dy, _dz
# ) where {T}
#     @inline d_xi(A) = _d_xi(A, _dx, I...)
#     @inline d_yi(A) = _d_yi(A, _dy, I...)
#     @inline d_zi(A) = _d_zi(A, _dz, I...)

#     if isvalid_c(ϕ, I...)
#         @inbounds ∇V[I...] = d_xi(Vx) + d_yi(Vy) + d_zi(Vz)
#     else
#         @inbounds ∇V[I...] = zero(T)
#     end
#     return nothing
# end

@parallel_indices (I...) function compute_∇V!(
    ∇V::AbstractArray{T,N}, V::NTuple{N}, ϕ::JustRelax.RockRatio, _di::NTuple{N}
) where {T,N}
    if isvalid_c(ϕ, I...)
        @inbounds ∇V[I...] = div(V..., _di..., I...)
    else
        @inbounds ∇V[I...] = zero(T)
    end
    return nothing
end

@parallel_indices (i, j) function compute_V!(
    Vx::AbstractArray{T,2},
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
            Vx[i + 1, j + 1] += R_Vx * ηdτ / av_xa(ητ, ϕ.center)
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
            Vy[i + 1, j + 1] += R_Vy * ηdτ / av_ya(ητ, ϕ.center)
        else
            Ry[i, j] = zero(T)
            Vy[i + 1, j + 1] = zero(T)
        end
    end

    return nothing
end
