
@parallel_indices (i, j) function compute_viscosity!(η, ν, εxx, εyy, εxyv, args, rheology)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)
   
    # we nee strain rate not to be zero, otherwise we get NaNs
    εII_0 = (εxx[i, j] == 0 && εyy[i, j] == 0) ? 1e-15 : 0.0

    # argument fields at local index
    v = getindex.(values(args), i, j)
    k = keys(args)
    args_ij  = (; zip(k, v)..., dt = args.dt, τII_old=0.0)

    # cache strain rate and stress 
    εij_p    = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
    τij_p_o  = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
    phases   = 1, 1, (1,1,1,1) # for now hard-coded for a single phase

    # update stress and effective viscosity
    _, _, ηi = compute_τij(rheology, εij_p, args_ij, τij_p_o, phases)
    ηi       = exp((1-ν)*log(η[i, j]) + ν*log(ηi))
    η[i, j]  = clamp(ηi, 1e16, 1e24)
    
    return nothing
end

@parallel_indices (i, j) function compute_viscosity!(η, ν, ratios_center, εxx, εyy, εxyv, args, MatParam)

    # convinience closure
    @inline gather(A) = _gather(A, i, j)
    @inline av(A)     = (A[i + 1, j] + A[i + 2, j] + A[i + 1, j + 1] + A[i + 2, j + 1]) * 0.25
   
    εII_0 = (εxx[i, j] == 0 && εyy[i, j] == 0) ? 1e-15 : 0.0
    @inbounds begin
        ratio_ij = ratios_center[i,j]
        args_ij  = (; dt = args.dt, P = (args.P[i, j]), T=av(args.T), τII_old=0.0)
        εij_p    = εII_0 + εxx[i, j], -εII_0 + εyy[i, j], gather(εxyv)
        τij_p_o  = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        # # update stress and effective viscosity
        _, _, ηi = compute_τij_ratio(MatParam, ratio_ij, εij_p, args_ij, τij_p_o)
        ηi       = exp((1-ν)*log(η[i, j]) + ν*log(ηi))
        η[i, j]  = clamp(ηi, 1e16, 1e24)
    end
    
    return nothing
end

