@parallel_indices (I...) function update_PAD!(
    P,
    ResP,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::C,
    dt,
    r,
    θ_dτ,
    kwargs,
) where {N,C<:JustRelax.CellArray}

    K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    RP[I...], P[I...] = _update_PAD!(P[I...], ResP[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

function _update_PAD!(P, ResP, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    RP = ResP
    P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)

    return RP, P
end

@parallel_indices (I...) function compute_P_kernelAD!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::C,
    dt,
    r,
    θ_dτ,
    ::Nothing,
    ::Nothing,
) where {N,C<:JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    RP[I...], P[I...] = _compute_PAD!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end


function _compute_PAD!(P, P0, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    RP = fma(-(P - P0), _Kdt, -∇V)
    ψ = inv(inv(r / θ_dτ * η) + _Kdt)
    P = ((fma(P0, _Kdt, -∇V)) * ψ + P) / (1 + _Kdt * ψ)
    #P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)
    return RP, P
end
