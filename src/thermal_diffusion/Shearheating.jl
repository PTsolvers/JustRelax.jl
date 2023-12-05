@parallel_indices (I...) function compute_SH!(
    H,
    τ,
    τII,
    τ_old,
    τII_old,
    ε,
    phase_ratios, #phase_v is passed onto thermal solver temperature2center to average? will probably update with phase_ratios
    rheology,
    dt,
    )
    phase = @inbounds phase_ratios[I...]
    _Gdt = inv(fn_ratio(get_G, rheology, phase) * dt)

    τij, τij_o, εij = JustRelax.cache_tensors(τ, τ_old, ε, I...)
    τII[I...] = τII_ij = second_invariant(τij...)
    τII_old[I...] = τII_ij_old = second_invariant(τij_o...)

    εij_el = 0.5 * ((τII_ij - τII_ij_old) * _Gdt)

    H[I...] = compute_shearheating(rheology, phase_ratios, τij, εij, εij_el)
    return nothing
end





# @parallel (@idx ni) compute_SH!(
#     thermal.H,
#     @tensor_center(stokes.τ),
#     stokes.τ.II,
#     @tensor_center(stokes.τ_o),
#     stokes.τ_o.II,
#     @strain(stokes),
#     phase_ratios.center,
#     tupleize(rheology), # needs to be a tuple
#     dt,
# )
