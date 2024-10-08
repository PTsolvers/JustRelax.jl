function compute_melt_fraction!(ϕ, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...] = compute_meltfraction(rheology, args_ijk)
    return nothing
end

function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, phase_ratios, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(
    ϕ, phase_ratios, rheology, args
)
    args_ijk = ntuple_idx(args, I...)
    # ϕ[I...] = compute_meltfraction_ratio(@cell(phase_ratios[I...]), rheology, args_ijk)
    ϕ[I...] = fn_ratio(compute_meltfraction, rheology, @cell(phase_ratios[I...]), args_ijk)

    return nothing
end
