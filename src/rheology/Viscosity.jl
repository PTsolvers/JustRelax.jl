@inline function local_viscosity_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt=args.dt, τII_old=0.0)
    return local_args
end

@inline function local_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)...)
    return local_args
end

# 2D kernel
@parallel_indices (i, j) function compute_viscosity!(η, ν, εxx, εyy, εxyv, args, rheology)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = (εxx[i, j] == εyy[i, j] == 0) * 1e-15

        # argument fields at local index
        args_ij = local_args(args, i, j)

        # compute second invariant of strain rate tensor
        εij = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ij)
        ηi = continuation_log(ηi, η[i, j], ν)
        η[i, j] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end

@parallel_indices (i, j) function compute_viscosity!(η, ν, εII, args, rheology)

    @inbounds begin
        # argument fields at local index
        args_ij = local_args(args, i, j)

        # compute second invariant of strain rate tensor
        εII_ij = εII[i, j]

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII_ij, args_ij)
        ηi = continuation_log(ηi, η[i, j], ν)
        η[i, j] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end

#3D kernel
@parallel_indices (i, j, k) function compute_viscosity!(
    η, ν, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology
)

    # convinience closures
    @inline gather_yz(A) = _gather_yz(A, i, j, k)
    @inline gather_xz(A) = _gather_xz(A, i, j, k)
    @inline gather_xy(A) = _gather_xy(A, i, j, k)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = (εxx[i, j, k] == εyy[i, j, k] == εzz[i, j, k] == 0) * 1e-18

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, i, j, k)

        # compute second invariant of strain rate tensor
        εij_normal = εxx[i, j, k], εyy[i, j, k], εzz[i, j, k]
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        εII = second_invariant(εij...)

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ijk)
        ηi = continuation_log(ηi, η[i, j, k], ν)
        η[i, j, k] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_viscosity!(
    η, ν, εII, args, rheology
)

    @inbounds begin
        # # argument fields at local index
        args_ijk = local_viscosity_args(args, i, j, k)

        # compute second invariant of strain rate tensor
        εII_ij = εII[i,j,k]

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, εII_ij, args_ijk)
        ηi = continuation_log(ηi, η[i, j, k], ν)
        η[i, j, k] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end

@parallel_indices (i, j) function compute_viscosity!(
    η, ν, ratios_center, εxx, εyy, εxyv, args, rheology
)
    # convinience closure
    @inline gather(A) = _gather(A, i, j)

    @inbounds begin
        # we nee strain rate not to be zero, otherwise we get NaNs
        εII_0 = (εxx[i, j] == εyy[i, j] == 0) * 1e-15

        # local phase ratio
        ratio_ij = ratios_center[i, j]

        # argument fields at local index
        args_ij = local_args(args, i, j)

        # compute second invariant of strain rate tensor
        εij = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, ratio_ij, εII, args_ij)
        ηi = continuation_log(ηi, η[i, j], ν)
        η[i, j] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end
