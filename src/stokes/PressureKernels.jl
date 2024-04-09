
"""
    init_P!(P, ρg, z)

Initialize the pressure field `P` based on the buoyancy forces `ρg` and the vertical coordinate `z`.

# Arguments
- `P::Array`: Pressure field to be initialized.
- `ρg::Float64`: Buoyancy forces.
- `z::Array`: Vertical coordinate.
"""
function init_P!(P, ρg, z)
    ni = size(P)
    @parallel (@idx ni) init_P_kernel!(P, ρg, z)
    return nothing
end

@parallel_indices (I...) function init_P_kernel!(P, ρg, z)
    P[I...] = abs(ρg[I...] * z[I[end]]) * (z[I[end]] < 0.0)
    return nothing
end

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
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase, dt, r, θ_dτ, args
) where {N}
    K = get_bulk_modulus(rheology, phase[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

"""
   compute_P!(P, P0, RP, ∇V, ΔTc, η, rheology::NTuple{N,MaterialParams}, phase_ratio::C, dt, r, θ_dτ)

Compute the pressure field `P` and the residual `RP` for the compressible case. This function introduces thermal stresses
after the implementation of Kiss et al. (2023). The temperature difference `ΔTc` on the cell center is used to compute this
as well as α as the thermal expansivity.
"""
function compute_P!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio,
    dt,
    r,
    θ_dτ,
    kwargs::NamedTuple,
) where {N}
    return compute_P!(P, P0, RP, ∇V, η, rheology, phase_ratio, dt, r, θ_dτ; kwargs...)
end

function compute_P!(
    P,
    P0,
    RP,
    ∇V,
    η,
    rheology::NTuple{N,MaterialParams},
    phase_ratio,
    dt,
    r,
    θ_dτ;
    ΔTc=nothing,
    kwargs...,
) where {N}
    ni = size(P)
    @parallel (@idx ni) compute_P_kernel!(
        P, P0, RP, ∇V, η, rheology, phase_ratio, dt, r, θ_dτ, ΔTc
    )
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
    dt,
    r,
    θ_dτ,
    ::Nothing,
) where {N,C<:JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, phase_ratio[I...])
    RP[I...], P[I...] = _compute_P!(P[I...], P0[I...], ∇V[I...], η[I...], K, dt, r, θ_dτ)
    return nothing
end

@parallel_indices (I...) function compute_P_kernel!(
    P, P0, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase_ratio::C, dt, r, θ_dτ, ΔTc
) where {N,C<:JustRelax.CellArray}
    K = fn_ratio(get_bulk_modulus, rheology, phase_ratio[I...])
    α = fn_ratio(get_thermal_expansion, rheology, phase_ratio[I...])
    RP[I...], P[I...] = _compute_P!(
        P[I...], P0[I...], ∇V[I...], ΔTc[I...], α, η[I...], K, dt, r, θ_dτ
    )
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

function _compute_P!(P, P0, ∇V, ΔTc, α, η, K, dt, r, θ_dτ)
    _Kdt = inv(K * dt)
    _dt = inv(dt)
    RP = fma(-(P - P0), _Kdt, (-∇V + (α * (ΔTc * _dt))))
    P += RP / (1.0 / (r / θ_dτ * η) + 1.0 * _Kdt)
    return RP, P
end
