function compute_P!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::JustPIC.PhaseRatios,
    ϕ::JustRelax.RockRatio,
    dt,
    r,
    θ_dτ;
    ΔTc=nothing,
    kwargs...,
) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_P_kernel!(
        P, P0, RP, ∇V, η, rheology, phase_ratio.center, ϕ, dt, r, θ_dτ, ΔTc
    )
    return nothing
end

@parallel_indices (I...) function compute_P!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio,
    ϕ::JustRelax.RockRatio,
    dt,
    r,
    θ_dτ,
    ::Nothing,
) where {N}
    if isvalid_c(ϕ, I...)
        K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
        RP[I...], P[I...] = _compute_P!(
            P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ
        )
    else
        RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio::C,
    ϕ::JustRelax.RockRatio,
    dt,
    r,
    θ_dτ,
    ΔTc,
    ::Nothing,
) where {N,C<:JustRelax.CellArray}
    if isvalid_c(ϕ, I...)
        phase_ratio_I = phase_ratio[I...]
        K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
        α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
        RP[I...], P[I...] = _compute_P!(
            P[I...], P0[I...], ∇V[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
        )
    else
        RP[I...] = P[I...] = zero(eltype(P))
    end
    return nothing
end
