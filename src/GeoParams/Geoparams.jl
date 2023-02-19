# BUOYANCY FORCE KERNELS
"""
    @parallel_indices (i, j) compute_ρg!(ρg, rheology, phase_c, args)

ParallelStencil kernel to compute the buoyancy force term `ρg`
"""
@parallel_indices (i, j) function compute_ρg!(ρg, rheology, args)
    
    av(A) = 0.25 * (A[i,j] + A[i+1,j] + A[i,j+1] + A[i+1,j+1])

    @inbounds ρg[i, j] =
        compute_density(rheology, (T=av(args.T), P=args.P[i,j])) * compute_gravity(rheology)
    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg, rheology, args)
    @inbounds ρg[i, j, k] =
        compute_density(rheology, ntuple_idx(args, i, j, k)) * compute_gravity(rheology)
    return nothing
end

"""
    @parallel_indices (i, j) compute_ρg!(ρg, rheology, phase_c, args)

ParallelStencil kernel to compute the buoyancy force `ρg` term for multiple rheology phases
"""
@parallel_indices (i, j) function compute_ρg!(ρg::A, rheology, phase_c, args) where A<:AbstractArray{_T,2} where _T
    
    av(A) = 0.25 * (A[i,j] + A[i+1,j] + A[i,j+1] + A[i+1,j+1])

    @inbounds phase = phase_c[i, j]
    @inbounds ρg[i, j] =
        compute_density(rheology, phase, (T=av(args.T), P=args.P[i,j])) * compute_gravity(rheology, phase)
    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg::A, rheology, phase_c, args) where A<:AbstractArray{_T,3} where _T
    @inbounds phase = phase_c[i, j, k]
    @inbounds ρg[i, j, k] =
        compute_density(rheology, phase, ntuple_idx(args, i, j, k)) *
        compute_gravity(rheology, phase)
    return nothing
end

# MELT FRACTION KERNELS
"""
    @parallel_indices (i, j) compute_melt_fraction!(ϕ, rheology, phase_c, args)

ParallelStencil kernel to compute the meltfraction `ϕ` for multiple rheology phases
"""
@parallel_indices (i, j) function compute_melt_fraction!(ϕ::A, rheology, phase_c, args) where A<:AbstractArray{Any,2}
    
    av() = 0.25 * sumt(ntuple_idx(args, i, j), ntuple_idx(args, i, j+1), ntuple_idx(args, i+1, j), ntuple_idx(args, i+1, j+1))
    
    @inbounds ϕ[i, j] = compute_meltfraction(rheology, phase_c[i, j], av())
    return nothing
end

@parallel_indices (i, j, k) function compute_melt_fraction!(ϕ::A, rheology, phase_c, args) where A<:AbstractArray{T,3} where T
    @inbounds ϕ[i, j, k] = compute_meltfraction(rheology, phase_c[i, j, k], ntuple_idx(args, i, j, k))
    return nothing
end


