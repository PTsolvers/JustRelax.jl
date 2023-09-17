## 2D KERNELS

@parallel_indices (I...) function compute_viscosity!(
    η, ν, εxx, εyy, εxyv, args, rheology, cutoff
)

    # convinience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = δ(εxx[I...], εyy[I...], 0.0) * 1e-15

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εij = εII_0 + εxx[I...], -εII_0 + εyy[I...], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@parallel_indices (I...) function compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    @inbounds begin
        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εII_ij = εII[I...]

        # compute and update stress viscosity
        ηi = compute_viscosity_εII(rheology, εII_ij, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

# function compute_viscosity!(η, x...)
#     ni = size(η)
#     @parallel (JustRelax.@idx ni) _compute_viscosity!(η, x...)
#     return nothing
# end

@parallel_indices (I...) function compute_viscosity!(
    η, ν, ratios_center, εxx, εyy, εxyv, args, rheology, cutoff
)

    # convinience closure
    @inline gather(A) = _gather(A, I...)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = δ(εxx[I...], εyy[I...], 0.0) * 1e-18

        # argument fields at local index
        args_ij = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ij = ratios_center[I...]

        # compute second invariant of strain rate tensor
        εij = εII_0 + εxx[I...], -εII_0 + εyy[I...], gather(εxyv)
        εII = second_invariant(εij...)

        # compute and update stress viscosity
        ηi = compute_phase_viscosity_εII(rheology, ratio_ij, εII, args_ij)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## 3D KERNELS

@parallel_indices (I...) function compute_viscosity!(
    η, ν, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology, cutoff
)

    # convinience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = δ(εxx[I...], εyy[I...], εzz[I...],0.0) * 1e-18

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εij_normal = εxx[I...], εyy[I...], εzz[I...]
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        εII = second_invariant(εij...)

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, εII, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@parallel_indices (I...) function compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    @inbounds begin
        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # compute second invariant of strain rate tensor
        εII_ij = εII[I...]

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, εII_ij, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

@parallel_indices (I...) function compute_viscosity!(
    η, ν, ratios_center, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology, cutoff
)

    # convinience closures
    @inline gather_yz(A) = _gather_yz(A, I...)
    @inline gather_xz(A) = _gather_xz(A, I...)
    @inline gather_xy(A) = _gather_xy(A, I...)

    @inbounds begin
        # we need strain rate not to be zero, otherwise we get NaNs
        εII_0 = δ(εxx[I...], εyy[I...], εzz[I...], 0.0) * 1e-18

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, I...)

        # local phase ratio
        ratio_ijk = ratios_center[I...]

        # compute second invariant of strain rate tensor
        εij_normal = εxx[I...], εyy[I...], εzz[I...]
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        εII = second_invariant(εij...)

        # update stress and effective viscosity
        ηi = compute_viscosity_εII(rheology, ratio_ijk, εII, args_ijk)
        ηi = continuation_log(ηi, η[I...], ν)
        η[I...] = clamp(ηi, cutoff...)
    end

    return nothing
end

## HELPER FUNCTIONS

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

@generated function compute_phase_viscosity_εII(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, εII, args
) where {N}
    quote
        Base.@_inline_meta
        η = 0.0
        Base.@nexprs $N i -> (
            η += if iszero(ratio[i])
                0.0
            else
                inv(compute_viscosity_εII(rheology[i].CompositeRheology[1], εII, args)) * ratio[i]
            end
        )
        inv(η)
    end
end
