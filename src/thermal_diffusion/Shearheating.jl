@parallel_indices (I...) function compute_SH!(
    H,
    τ,
    τII,
    τ_old,
    τII_old,
    ε,
    phase_ratios,
    rheology,
    dt,
    )

    phase = @inbounds phase_ratios[I...]
    _Gdt = inv(fn_ratio(get_G, rheology, phase) * dt)
    τij, τij_o, εij = JustRelax.cache_tensors(τ, τ_old, ε, I...)
    τII[I...] = τII_ij = second_invariant(τij...)
    τII_old[I...] = τII_ij_old = second_invariant(τij_o...)

    εij_el = 0.5 * ((τII_ij - τII_ij_old) * _Gdt)
    H[I...] = H_s = compute_shearheating(rheology, phase, τij, εij, εij_el) # @AdM - i'm not sure about the H[I...] = H_s part
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
#     rheology,
#     dt,
# )
