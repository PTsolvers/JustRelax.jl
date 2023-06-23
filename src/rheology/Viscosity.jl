@inline function local_viscosity_args(args, I::Vararg{Integer,N}) where {N}
    v = getindex.(values(args), I...)
    local_args = (; zip(keys(args), v)..., dt=args.dt, τII_old=0.0)
    return local_args
end

# 2D kernel
@parallel_indices (i, j) function compute_viscosity!(η, ν, εxx, εyy, εxyv, args, rheology)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)

    @inbounds begin
        # we nee strain rate not to be zero, otherwise we get NaNs
        εII_0 = (εxx[i, j] == εyy[i, j] == 0) * 1e-15

        # argument fields at local index
        args_ij = local_viscosity_args(args, i, j, k)

        # cache strain rate and stress 
        εij_p = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        τij_p_o = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases = 1, 1, (1, 1, 1, 1) # for now hard-coded for a single phase

        # update stress and effective viscosity
        _, _, ηi = compute_τij(rheology, εij_p, args_ij, τij_p_o, phases)
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

    _zeros = 0.0, 0.0, 0.0, 0.0
    _ones = 1, 1, 1, 1

    # we nee strain rate not to be zero, otherwise we get NaNs
    @inbounds begin
        εII_0 = (εxx[i, j, k] == εyy[i, j, k] == εzz[i, j, k] == 0) * 1e-18

        # # argument fields at local index
        args_ijk = local_viscosity_args(args, i, j, k)

        # cache strain rate and stress 
        εij_normal = εxx[i, j, k], εyy[i, j, k], εzz[i, j, k]
        εij_normal = εij_normal .+ (εII_0, -εII_0 * 0.5, -εII_0 * 0.5)
        εij_shear = gather_yz(εyzv), gather_xz(εxzv), gather_xy(εxyv)
        εij = (εij_normal..., εij_shear...)
        τij_o = 0.0, 0.0, 0.0, _zeros, _zeros, _zeros
        phases = 1, 1, 1, _ones, _ones, _ones

        # update stress and effective viscosity
        _, _, ηi = compute_τij(rheology, εij, args_ijk, τij_o, phases)
        ηi = continuation_log(ηi, η[i, j, k], ν)
        η[i, j, k] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end

@parallel_indices (i, j) function compute_viscosity!(
    η, ν, ratios_center, εxx, εyy, εxyv, args, MatParam
)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)
    @inline av(A) = (A[i + 1, j] + A[i + 2, j] + A[i + 1, j + 1] + A[i + 2, j + 1]) * 0.25

    εII_0 = (εxx[i, j] == εyy[i, j] == 0) ? 1e-15 : 0.0
    @inbounds begin
        ratio_ij = ratios_center[i, j]
        args_ij = (; dt=args.dt, P=(args.P[i, j]), T=av(args.T), τII_old=0.0)
        εij_p = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        τij_p_o = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        # # update stress and effective viscosity
        _, _, ηi = compute_τij_ratio(MatParam, ratio_ij, εij_p, args_ij, τij_p_o)
        ηi = exp((1 - ν) * log(η[i, j]) + ν * log(ηi))
        η[i, j] = clamp(ηi, 1e16, 1e24)
    end

    return nothing
end
