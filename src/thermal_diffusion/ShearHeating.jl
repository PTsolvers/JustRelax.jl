"""
    compute_shear_heating!(thermal, stokes, rheology, dt)

Populate `thermal.shear_heating` from the current deviatoric stress and strain-rate
fields stored in `stokes`.

The kernel evaluates the elastic strain-rate contribution from `stokes.τ`,
`stokes.τ_o`, the shear modulus in `rheology`, and the pseudo-time step `dt`.
The resulting volumetric heating term is written in place on thermal cell centers.

When phase ratios are passed as an extra positional argument, phase-weighted
material properties are used instead of a single rheology state.
"""
function compute_shear_heating!(thermal, args...)
    return compute_shear_heating!(backend(thermal), thermal, args...)
end

function compute_shear_heating!(::CPUBackendTrait, thermal, stokes, rheology, dt)
    ni = size(thermal.shear_heating)
    @parallel (ni) compute_shear_heating_kernel!(
        thermal.shear_heating,
        @tensor_center(stokes.τ),
        @tensor_center(stokes.τ_o),
        @strain(stokes),
        rheology,
        dt,
    )
    return nothing
end

@parallel_indices (I...) function compute_shear_heating_kernel!(
        shear_heating, τ::NTuple{N, T}, τ_old::NTuple{N, T}, ε::NTuple{N, T}, rheology, dt
    ) where {N, T}
    _Gdt = inv(get_shear_modulus(rheology) * dt)
    τij, τij_o, εij = cache_tensors(τ, τ_old, ε, I...)
    εij_el = @. 0.5 * ((τij - τij_o) * _Gdt)
    shear_heating[I...] = compute_shearheating(rheology, τij, εij, εij_el)
    return nothing
end

function compute_shear_heating!(
        ::CPUBackendTrait, thermal, stokes, phase_ratios::JustPIC.PhaseRatios, rheology, dt
    )
    ni = size(thermal.shear_heating)
    @parallel (@idx ni) compute_shear_heating_kernel!(
        thermal.shear_heating,
        @tensor_center(stokes.τ),
        @tensor_center(stokes.τ_o),
        @strain(stokes),
        phase_ratios.center,
        rheology,
        dt,
    )
    return nothing
end

@parallel_indices (I...) function compute_shear_heating_kernel!(
        shear_heating,
        τ::NTuple{N, T},
        τ_old::NTuple{N, T},
        ε::NTuple{N, T},
        phase_ratios::CellArray,
        rheology,
        dt,
    ) where {N, T}
    phase = @inbounds phase_ratios[I...]
    _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
    τij, τij_o, εij = cache_tensors(τ, τ_old, ε, I...)
    εij_el = @. 0.5 * ((τij - τij_o) * _Gdt)
    shear_heating[I...] = max(0.0e0, compute_shearheating(rheology, phase, τij, εij, εij_el))
    return nothing
end
