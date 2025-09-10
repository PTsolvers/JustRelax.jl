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
    dτPt,
    kwargs,
) where {N,C<:JustRelax.CellArray}

    #K    = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    #_Kdt = inv(K * dt)
    #P[I...] = P[I...] + ResP[I...] / (1.0 / (r / θ_dτ * η[I...]) + 1.0 * _Kdt)
    #P[I...] = P[I...] + ResP[I...]*r / θ_dτ * η[I...]
    P[I...] = P[I...] + ResP[I...]* dτPt[I...]
    return nothing
end

#=
function _update_PAD!(P, ResP, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    P += ResP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)

    return ResP, P
end
=#

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

@parallel_indices (I...) function compute_P_kernelADSens!(
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
    Sens,
    ::Nothing,
    ::Nothing,
) where {N,C<:JustRelax.CellArray}
    #K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    K = Sens[4][I...]
    RP[I...], P[I...] = _compute_PAD!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end


function _compute_PAD!(P, P0, ∇V, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    RP = fma(-(P - P0), _Kdt, -∇V)
    #ψ = inv(inv(r / θ_dτ * η) + _Kdt)
    #P = ((fma(P0, _Kdt, -∇V)) * ψ + P) / (1 + _Kdt * ψ)
    #P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)
    return RP, P
end

