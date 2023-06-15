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
        η[i, j]  = clamp(2*ηi, 1e16, 1e24)
    end
    
    return nothing
end