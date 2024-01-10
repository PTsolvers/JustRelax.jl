@parallel_indices (I...) function compute_shear_heating!(
    shear_heating, τ::NTuple{N,T}, τ_old::NTuple{N,T}, ε::NTuple{N,T}, rheology, dt
) where {N,T}
    _Gdt = inv(get_G(rheology) * dt)
    τij, τij_o, εij = cache_tensors(τ, τ_old, ε, I...)
    εij_el = @. 0.5 * ((τij - τij_o) * _Gdt)
    shear_heating[I...] = compute_shearheating(rheology, τij, εij, εij_el)
    return nothing
end

@parallel_indices (I...) function compute_shear_heating!(
    shear_heating,
    τ::NTuple{N,T},
    τ_old::NTuple{N,T},
    ε::NTuple{N,T},
    phase_ratios::CellArray,
    rheology,
    dt,
) where {N,T}
    phase = @inbounds phase_ratios[I...]
    _Gdt = inv(fn_ratio(get_G, rheology, phase) * dt)
    τij, τij_o, εij = cache_tensors(τ, τ_old, ε, I...)
    εij_el = @. 0.5 * ((τij - τij_o) * _Gdt)
    shear_heating[I...] = compute_shearheating(rheology, phase, τij, εij, εij_el)
    return nothing
end
