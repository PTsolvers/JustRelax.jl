"""
    compute_melt_fraction!(ϕ, rheology, args)
    compute_melt_fraction!(ϕ, phase_ratios::JustPIC.PhaseRatios, rheology, args)

Fill the melt-fraction array `ϕ` from the GeoParams melting closures in
`rheology`. The phase-ratios method combines per-phase melt fractions with
`fn_ratio`. `args` supplies `T` and `P` (scalars or index-matched arrays).
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
