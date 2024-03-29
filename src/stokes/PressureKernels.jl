# Continuity equation

## Incompressible 
@parallel_indices (I...) function compute_P!(P, RP, ∇V, η, r, θ_dτ)
    RP[I...], P[I...] = _compute_P!(P[I...], ∇V[I...], η[I...], r, θ_dτ)
    return nothing
end

## Compressible 
@parallel_indices (I...) function compute_P!(P, P0, RP, ∇V, η, K, dt, r, θ_dτ)
    RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], η[I...], K[I...], dt, r, θ_dτ
    )
    return nothing
end

# With GeoParams

@parallel_indices (I...) function compute_P!(
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase, dt, r, θ_dτ
) where {N}
    K = get_bulk_modulus(rheology, phase[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (I...) function compute_P!(
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase_ratio::C, dt, r, θ_dτ
) where {N,C<:JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, phase_ratio[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

# Pressure innermost kernels 

function _compute_P!(P, ∇V, η, r, θ_dτ)
    RP = -∇V
    P += RP * r / θ_dτ * η
    return RP, P
end

function _compute_P!(P, P0, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    RP = fma(-(P - P0), _Kdt, -∇V)
    P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)
    return RP, P
end
