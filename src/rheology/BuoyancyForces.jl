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

"""
    compute_buoyancy(rheology::MaterialParams, args)

Compute the buoyancy forces based on the given rheology parameters and arguments.

# Arguments
- `rheology::MaterialParams`: The material parameters for the rheology.
- `args`: The arguments for the computation.
"""
@inline function compute_buoyancy(rheology::MaterialParams, args)
    return compute_density(rheology, args) * compute_gravity(rheology)
end

"""
    compute_buoyancy(rheology::MaterialParams, args, phase_ratios)

Compute the buoyancy forces for a given set of material parameters, arguments, and phase ratios.

# Arguments
- `rheology`: The material parameters.
- `args`: The arguments.
- `phase_ratios`: The phase ratios.
"""
@inline function compute_buoyancy(rheology::MaterialParams, args, phase_ratios)
    return compute_density_ratio(phase_ratios, rheology, args) * compute_gravity(rheology)
end

"""
    compute_buoyancy(rheology, args)

Compute the buoyancy forces based on the given rheology and arguments.

# Arguments
- `rheology`: The rheology used to compute the buoyancy forces.
- `args`: Additional arguments required for the computation.
"""
@inline function compute_buoyancy(rheology, args)
    return compute_density(rheology, args) * compute_gravity(rheology[1])
end

"""
    compute_buoyancy(rheology, args, phase_ratios)

Compute the buoyancy forces based on the given rheology, arguments, and phase ratios.

# Arguments
- `rheology`: The rheology used to compute the buoyancy forces.
- `args`: Additional arguments required by the rheology.
- `phase_ratios`: The ratios of the different phases.
"""
@inline function compute_buoyancy(rheology, args, phase_ratios)
    return fn_ratio(compute_density, rheology, phase_ratios, args) *
           compute_gravity(rheology[1])
end
