"""
    compute_ρg!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""

@parallel_indices (I...) function compute_ρg!(ρg, rheology, args)   # index arguments for the current cell cell center
    args_ijk = ntuple_idx(args, I...)
    ρg[I...] = JustRelax.compute_buoyancy(rheology, args_ijk)
    return nothing
end

"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
The `phase_ratios` are used to compute the density of the composite rheology.
"""
@parallel_indices (I...) function compute_ρg!(ρg, phase_ratios, rheology, args)   # index arguments for the current cell cell center
    args_ijk = ntuple_idx(args, I...)
    ρg[I...] = JustRelax.compute_buoyancy(rheology, args_ijk, phase_ratios[I...])
    return nothing
end

## Inner buoyancy force kernels

@inline function compute_buoyancy(rheology::MaterialParams, args)
    return compute_density(rheology, args) * compute_gravity(rheology)
end

@inline function compute_buoyancy(rheology::MaterialParams, args, phase_ratios)
    return compute_density_ratio(phase_ratios, rheology, args) * compute_gravity(rheology)
end

@inline function compute_buoyancy(rheology, args)
    return compute_density(rheology, args) * compute_gravity(rheology[1])
end

# @inline function compute_buoyancy(rheology, args, phase_ratios)
#     return compute_density_ratio(phase_ratios, rheology, args) *
#            compute_gravity(rheology[1])
# end

@inline function compute_buoyancy(rheology, args, phase_ratios)
    return fn_ratio(compute_density, rheology, phase_ratios, args) *
           compute_gravity(rheology[1])
end
