"""
    compute_ρg!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""
@parallel_indices (i, j) function compute_ρg!(
    ρg::_T, rheology, args
) where {_T<:AbstractArray{M,2} where {M<:Real}}
    _compute_ρg!(ρg, rheology, args, i, j)
    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(
    ρg::_T, rheology, args
) where {_T<:AbstractArray{M,3} where {M<:Real}}
    _compute_ρg!(ρg, rheology, args, i, j, k)
    return nothing
end

function _compute_ρg!(ρg, rheology, args, I::Vararg{Int,N}) where {N}
    # index arguments for the current cell cell center
    T = args.T[I...] - 273.0
    P = args.P[I...]
    args_ijk = (; T, P)
    return ρg[I...] = compute_buoyancy(rheology, args_ijk)
end

"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
The `phase_ratios` are used to compute the density of the composite rheology.
"""
@parallel_indices (i, j) function compute_ρg!(
    ρg::_T, phase_ratios, rheology, args
) where {_T<:AbstractArray{M,2} where {M<:Real}}
    _compute_ρg!(ρg, phase_ratios, rheology, args, i, j)
    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(
    ρg::_T, phase_ratios, rheology, args
) where {_T<:AbstractArray{M,3} where {M<:Real}}
    _compute_ρg!(ρg, phase_ratios, rheology, args, i, j, k)
    return nothing
end

function _compute_ρg!(ρg, phase_ratios, rheology, args, I::Vararg{Int,N}) where {N}
    # index arguments for the current cell cell center
    T = args.T[I...] - 273.0
    P = args.P[I...]
    args_ijk = (; T, P)
    return ρg[I...] = compute_buoyancy(rheology, args_ijk, phase_ratios[I...])
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

@inline function compute_buoyancy(rheology, args, phase_ratios)
    return compute_density_ratio(phase_ratios, rheology, args) *
           compute_gravity(rheology[1])
end
