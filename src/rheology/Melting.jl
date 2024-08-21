function compute_melt_fraction!(ϕ, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...] = GeoParams.compute_meltfraction(rheology, args_ijk)
    return nothing
end

function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
    ni = size(ϕ)
    @parallel (@idx ni) compute_melt_fraction_kernel!(ϕ, phase_ratios, rheology, args)
end

@parallel_indices (I...) function compute_melt_fraction_kernel!(ϕ, phase_ratios, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ϕ[I...]  = GeoParams.compute_meltfraction_ratio(rheology, args_ijk, phase_ratios[I...])
    # ϕ[I...] = compute_melt_frac(rheology, (;T=args.T[I...]), phase_ratios[I...])
    return nothing
end
