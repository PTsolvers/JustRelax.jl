# Pressure innermost kernels 

function _compute_P!(P, ∇V, η, r, θ_dτ)
    RP = -∇V
    P += RP * r / θ_dτ * η
    return RP, P
end

function _compute_P!(P, P0, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    RP = -∇V - (P - P0) * _Kdt
    P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)
    return RP, P
end

# Continuity equation

## Incompressible 
@parallel_indices (i, j) function compute_P!(
    P::AbstractArray{T,2}, RP, ∇V, η, r, θ_dτ
) where {T}
    I = i, j
    RP[I...], P[I...] = _compute_P!(P[I...], ∇V[I...], η[I...], r, θ_dτ)
    return nothing
end

@parallel_indices (i, j, k) function compute_P!(
    P::AbstractArray{T,3}, RP, ∇V, η, r, θ_dτ
) where {T}
    I = i, j, k
    RP[I...], P[I...] = _compute_P!(P[I...], ∇V[I...], η[I...], r, θ_dτ)
    return nothing
end

## Compressible 
@parallel_indices (i, j) function compute_P!(
    P::AbstractArray{T,2}, P0, RP, ∇V, η, K, dt, r, θ_dτ
) where {T}
    I = i, j
    RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], η[I...], K[I...], dt, r, θ_dτ
    )
    return nothing
end

@parallel_indices (i, j, k) function compute_P!(
    P::AbstractArray{T,3}, P0, RP, ∇V, η, K, dt, r, θ_dτ
) where {T}
    I = i, j, k
    RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], η[I...], K[I...], dt, r, θ_dτ
    )
    return nothing
end

# With GeoParams

@parallel_indices (i, j) function compute_P!(
    P::AbstractArray{T,2},
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase,
    dt,
    r,
    θ_dτ,
) where {T,N}
    I = i, j
    K = get_Kb(rheology, phase[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (i, j, k) function compute_P!(
    P::AbstractArray{T,3},
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase,
    dt,
    r,
    θ_dτ,
) where {T,N}
    I = i, j, k
    K = get_Kb(rheology, phase[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (i, j) function compute_P!(
    P::AbstractArray{T,2},
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::C,
    dt,
    r,
    θ_dτ,
) where {N,T,C<:JustRelax.CellArray}
    I = i, j
    K = fn_ratio(get_Kb, rheology, phase_ratio[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (i, j, k) function compute_P!(
    P::AbstractArray{T,3},
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::C,
    dt,
    r,
    θ_dτ,
) where {N,T,C<:JustRelax.CellArray}
    I = i, j, k
    K = fn_ratio(get_Kb, rheology, phase_ratio[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end
