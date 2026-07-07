function compute_residual_P!(
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
    return compute_residual_P!(P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio, dt; kwargs...)
end

function compute_residual_P!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        dt;
        ΔT = nothing,
        melt_fraction = nothing,
        kwargs...,
    ) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_RP_kernel!(
        P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio.center, dt, ΔT, melt_fraction
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
    RP[I...] = _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
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
    RP[I...] = _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
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
        ΔT,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    phase_ratio_I = phase_ratio[I...]
    α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
    RP[I...] = _compute_RP!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I .+ 1...], α, ηb[I...], dt
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
        ΔT,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
    RP[I...] = _compute_RP!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I .+ 1...], α, ηb[I...], dt
    )
    return nothing
end

# variational version: ηb = 0 in air ⇒ (P - P0)/ηb is NaN, so RP is masked to 0 there.
function compute_residual_P!(
        RP,
        P,
        P0,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        ϕ::JustRelax.RockRatio,
        dt,
        kwargs::NamedTuple,
    ) where {N}
    return compute_residual_P!(P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio, ϕ, dt; kwargs...)
end

function compute_residual_P!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        ηb,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        ϕ::JustRelax.RockRatio,
        dt;
        ΔT = nothing,
        melt_fraction = nothing,
        kwargs...,
    ) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_RP_kernel!(
        P, P0, RP, ∇V, Q, ηb, rheology, phase_ratio.center, ϕ, dt, ΔT, melt_fraction
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
        ϕ::JustRelax.RockRatio,
        dt,
        ::Nothing,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    RP[I...] = if isvalid_c(ϕ, I...)
        _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
    else
        0.0e0
    end
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
        ϕ::JustRelax.RockRatio,
        dt,
        ::Nothing,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    RP[I...] = if isvalid_c(ϕ, I...)
        _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ηb[I...], dt)
    else
        0.0e0
    end
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
        ϕ::JustRelax.RockRatio,
        dt,
        ΔT,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    RP[I...] = if isvalid_c(ϕ, I...)
        phase_ratio_I = phase_ratio[I...]
        α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
        _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I .+ 1...], α, ηb[I...], dt)
    else
        0.0e0
    end
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
        ϕ::JustRelax.RockRatio,
        dt,
        ΔT,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    RP[I...] = if isvalid_c(ϕ, I...)
        α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
        _compute_RP!(P[I...], P0[I...], ∇V[I...], Q[I...], ΔT[I .+ 1...], α, ηb[I...], dt)
    else
        0.0e0
    end
    return nothing
end

@inline _compute_RP!(P, P0, ∇V, Q, ηb, dt) = -∇V - (P - P0) / ηb + (Q / dt)
@inline _compute_RP!(P, P0, ∇V, Q, ΔT, α, ηb, dt) = -∇V - (P - P0) / ηb + α * (ΔT / dt) + (Q / dt)

# Per-cell pressure residual, dispatching on the presence of ΔT / melt_fraction exactly as
# `compute_RP_kernel!` does. Used to fold the RP computation into the strain-rate kernel while
# reusing `_compute_RP!` and keeping the thermal branch resolved at compile time (GPU-friendly).
@inline _RP_cell(P, P0, ∇V, Q, ηb, dt, rheology, phase_ratio, ::Nothing, ::Nothing, I::Vararg{Int}) =
    _compute_RP!(P, P0, ∇V, Q, ηb, dt)
@inline _RP_cell(P, P0, ∇V, Q, ηb, dt, rheology, phase_ratio, ::Nothing, melt_fraction, I::Vararg{Int}) =
    _compute_RP!(P, P0, ∇V, Q, ηb, dt)
@inline function _RP_cell(P, P0, ∇V, Q, ηb, dt, rheology, phase_ratio, ΔT, ::Nothing, I::Vararg{Int})
    α = fn_ratio(get_thermal_expansion, rheology, phase_ratio[I...])
    return _compute_RP!(P, P0, ∇V, Q, ΔT[(I .+ 1)...], α, ηb, dt)
end
@inline function _RP_cell(P, P0, ∇V, Q, ηb, dt, rheology, phase_ratio, ΔT, melt_fraction, I::Vararg{Int})
    α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
    return _compute_RP!(P, P0, ∇V, Q, ΔT[(I .+ 1)...], α, ηb, dt)
end
