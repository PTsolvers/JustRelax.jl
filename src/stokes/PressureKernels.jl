# Continuity equation

## Incompressible
@parallel_indices (I...) function compute_P!(P, RP, ∇V, Q, η, dt, r, θ_dτ)
    @inbounds RP[I...], P[I...] = _compute_P!(P[I...], ∇V[I...], Q[I...], η[I...], dt, r, θ_dτ)
    return nothing
end

## Compressible
@parallel_indices (I...) function compute_P!(P, P0, RP, ∇V, Q, η, K, G, dt, r, θ_dτ)
    @inbounds RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K[I...], G[I...], dt, r, θ_dτ
    )
    return nothing
end

  # With GeoParams

@parallel_indices (I...) function compute_P!(
        P, P0, RP, ∇V, Q, η, rheology::NTuple{N, MaterialParams}, phase, dt, r, θ_dτ, args
    ) where {N}
    K = get_bulk_modulus(rheology, phase[I...])
    G = get_shear_modulus(rheology, phase[I...])
    @inbounds RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K, G, dt, r, θ_dτ)
    return nothing
end

"""
   compute_P!(P, P0, RP, ∇V, Q, ΔTc, η, rheology::NTuple{N,MaterialParams}, phase_ratio::C, dt, r, θ_dτ)

Compute the pressure field `P` and the residual `RP` for the compressible case. This function introduces thermal stresses after the implementation of Kiss et al. (2023).

## Arguments
- `P`: pressure field
- `RP`: residual field
- `∇V`: divergence of the velocity field
- `Q`: volumetric source/sink term which should have the properties of `dV/V_tot [m³/m³]` normalized per cell, default is zero.
- `ΔTc`: temperature difference on the cell center, to account for thermal stresses. The thermal expansivity `α` is computed from the material parameters.
- `η`: viscosity field
- `rheology`: material parameters
- `phase_ratio`: phase field
"""
function compute_P!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        dt,
        r,
        θ_dτ,
        kwargs::NamedTuple,
    ) where {N}
    return compute_P!(P, P0, RP, ∇V, Q, η, rheology, phase_ratio, dt, r, θ_dτ; kwargs...)
end

function compute_P!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio,
        dt,
        r,
        θ_dτ;
        ΔTc = nothing,
        melt_fraction = nothing,
        kwargs...,
    ) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_P_kernel!(
        P, P0, RP, ∇V, Q, η, rheology, phase_ratio.center, dt, r, θ_dτ, ΔTc, melt_fraction
    )
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        r,
        θ_dτ,
        ::Nothing,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    G = fn_ratio(get_shear_modulus, rheology, @cell(phase_ratio[I...]))
    @inbounds RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K, G, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        r,
        θ_dτ,
        ::Nothing,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    G = fn_ratio(get_shear_modulus, rheology, @cell(phase_ratio[I...]))
    @inbounds RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], Q[I...], η[I...], K, G, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        r,
        θ_dτ,
        ΔTc,
        ::Nothing,
    ) where {N, C <: JustRelax.CellArray}
    @inbounds phase_ratio_I = phase_ratio[I...]
    @inbounds K = fn_ratio(get_bulk_modulus, rheology, phase_ratio_I)
    @inbounds G = fn_ratio(get_shear_modulus, rheology, phase_ratio_I)
    @inbounds α = fn_ratio(get_thermal_expansion, rheology, phase_ratio_I)
    @inbounds RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, η[I...], K, G, dt, r, θ_dτ
    )
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
        P,
        P0,
        RP,
        ∇V,
        Q, # volumetric source/sink term
        η,
        rheology::NTuple{N, MaterialParams},
        phase_ratio::C,
        dt,
        r,
        θ_dτ,
        ΔTc,
        melt_fraction,
    ) where {N, C <: JustRelax.CellArray}
    @inbounds K = fn_ratio(get_bulk_modulus, rheology, @cell(phase_ratio[I...]))
    @inbounds G = fn_ratio(get_shear_modulus, rheology, @cell(phase_ratio[I...]))
    @inbounds α = fn_ratio(get_thermal_expansion, rheology, @cell(phase_ratio[I...]), (; ϕ = melt_fraction[I...]))
    @inbounds RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], Q[I...], ΔTc[I...], α, η[I...], K, G, dt, r, θ_dτ
    )
    return nothing
end

# Pressure innermost kernels

function _compute_P!(P, ∇V, Q, η, dt, r, θ_dτ)
    RP = -∇V + (Q * inv(dt))
    P += RP * r / θ_dτ * η
    return RP, P
end

function _compute_P!(P, P0, ∇V, Q, η, K, G, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    _Gdt = inv(G * dt)
    _dt = inv(dt)
    RP = muladd(-(P - P0), _Kdt, (-∇V + (Q * _dt)))
    ψ = inv(inv(η) + _Gdt) * r / θ_dτ
    P = ((muladd(P0, _Kdt, (-∇V + (Q * _dt)))) * ψ + P) / (1 + _Kdt * ψ)

    return RP, P
end

function _compute_P!(P, P0, ∇V, Q, ΔTc, α, η, K, G, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    _Gdt = inv(G * dt)
    _dt = inv(dt)
    RP = muladd(-(P - P0), _Kdt, (-∇V + (α * (ΔTc * _dt)) + (Q * _dt)))
    ψ = inv(inv(η) + _Gdt) * r / θ_dτ
    P = ((muladd(P0, _Kdt, (-∇V + (α * (ΔTc * _dt)) + (Q * _dt)))) * ψ + P) / (1 + _Kdt * ψ)

    return RP, P
end
