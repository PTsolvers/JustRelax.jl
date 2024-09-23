function compute_melt_fraction!(ϕ, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...] = compute_melt_frac(rheology, args_ijk)
    return nothing
end

@inline function compute_melt_frac(rheology, args)
    return compute_meltfraction(rheology, args)
end

function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, phase_ratios, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(
    ϕ, phase_ratios, rheology, args
)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
    return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return compute_meltfraction_ratio(phase_ratios, rheology, args)
end
