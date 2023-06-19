"""
    compute_ρg!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""
@parallel_indices (i, j) function compute_ρg!(ρg::_T, rheology, args) where {_T<:AbstractArray{M, 2} where M<:Real}
     # index arguments for the current cell cell center
    T = args.T[i, j] - 273.0 
    P = args.P[i, j]
    args_ij =(; T, P)
    
    # compute ρg = density * gravity
    @inbounds ρg[i, j] = -compute_ρg(rheology, args_ij)

    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg::_T, rheology, args) where {_T<:AbstractArray{M, 3} where M<:Real}

    av_T() = _av(args.T, i, j, k) - 273.0

    # index arguments for the current cell cell center
    T = av_T()
    P = args.P[i, j, k]
    args_ijk =(; T, P)

    ρg[i, j, k] = compute_ρg(rheology, args_ijk)
    
    return nothing
end

"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
The `phase_ratios` are used to compute the density of the composite rheology.
"""
@parallel_indices (i, j) function compute_ρg!(ρg::_T, phase_ratios, rheology, args) where {_T<:AbstractArray{M, 2} where M<:Real}

    T = args.T[i, j] - 273.0 
    P = args.P[i, j]
    args_ij =(; T, P)

    # compute ρg = density * gravity
    @inbounds ρg[i, j] = -compute_ρg(rheology, args_ij, phase_ratios[i, j])

    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg::_T, phase_ratios, rheology, args) where {_T<:AbstractArray{M, 3} where M<:Real}

    av_T() = _av(args.T, i, j, k) - 273.0

    # index arguments for the current cell cell center
    T = av_T()
    P = args.P[i, j, k]
    args_ijk =(; T, P)

    ρg[i, j, k] = compute_ρg(rheology, args_ijk, phase_ratios[i, j, k])
    return nothing
end

@inline compute_ρg(rheology::MaterialParams, args) = compute_density(rheology, args) * compute_gravity(rheology)
@inline compute_ρg(rheology::MaterialParams, args, phase_ratios) = compute_density(phase_ratios, rheology, args) * compute_gravity(rheology)

@inline compute_ρg(rheology, args) = compute_density(rheology, args) * compute_gravity(rheology[1])
@inline compute_ρg(rheology, args, phase_ratios) = compute_density(phase_ratios, rheology, args) * compute_gravity(rheology[1])


