
function compute_RP!(
        RP,
        P,
        P0,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        dt,
        kwargs::NamedTuple,
    ) where {N}
    return compute_RP!(P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio, dt; kwargs...)
end

function compute_RP!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        dt;
        ΔTc = nothing,
        melt_fraction = nothing,
        kwargs...,
    ) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_RP_kernel!(
        P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio.center, dt, ΔTc, melt_fraction
    )
    return nothing
end

@parallel_indices (I...) function compute_RP_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        ::Nothing,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
    return nothing
end

@parallel_indices (I...) function compute_RP_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        ::Nothing,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
    return nothing
end

@parallel_indices (I...) function compute_RP_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        ΔTc,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    @inbounds phase_ratio_I = phase_ratio[I...]
    @inbounds α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
    @inbounds RP[I...] = _compute_RP!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, ηb[I...], dt
    )
    return nothing
end

@parallel_indices (I...) function compute_RP_kernel!(
        RP,
        P,
        P0,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        ΔTc,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    @inbounds α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
    @inbounds RP[I...] = _compute_RP!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, ηb[I...], dt
    )
    return nothing
end

@inline _compute_RP!(P, P0, ∇V, Q, ηb, dt) = -∇V - (P - P0) / ηb + (Q / dt)
@inline _compute_RP!(P, P0, ∇V, Q, ΔTc, α, ηb, dt) = -∇V - (P - P0) / ηb  + α * (ΔTc / dt) + (Q * inv(dt)) 
