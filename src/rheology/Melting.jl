"""
    compute_melt_fraction!(ϕ, rheology, args)
    compute_melt_fraction!(ϕ, phase_ratios::JustPIC.PhaseRatios, rheology, args)

Evaluate the melt fraction from the GeoParams `rheology` and the state fields in `args`
(e.g. `(; T, P)`), writing the result into `ϕ` in place. When `phase_ratios` are given, the
melt fraction is averaged over phases using the cell-center ratios.
"""
function compute_melt_fraction!(ϕ, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, rheology, args)
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, rheology, args)
    args_ijk = getindex_NamedTuple(args, I...)
    @inbounds ϕ[I...] = compute_meltfraction(rheology, args_ijk)
    return nothing
end

function compute_melt_fraction!(ϕ, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    ni = size(ϕ)
    return @parallel (@idx ni) compute_melt_fraction_kernel!(
        ϕ, phase_ratios.center, rheology, args
    )
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(
        ϕ, phase_ratios, rheology, args
    )
    args_ijk = getindex_NamedTuple(args, I...)
    @inbounds ϕ[I...] = fn_ratio(compute_meltfraction, rheology, @cell(phase_ratios[I...]), args_ijk)
    return nothing
end
