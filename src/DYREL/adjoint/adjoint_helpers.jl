function initialize_adjoint_iteration!(adjoint, ni)
    @parallel (@idx ni .+ 2) _initialize_adjoint_iteration!(
        adjoint.P,
        (adjoint.V.Vx, adjoint.V.Vy),
        (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy),
        (adjoint.R.Rx, adjoint.R.Ry, adjoint.R.RP),
        (adjoint.λV.Vx, adjoint.λV.Vy),
        adjoint.λP,
    )
    return nothing
end

@parallel_indices (i, j) function _initialize_adjoint_iteration!(
        P, V, ε, τ, R, λV, λP
    )
    if i ≤ size(P, 1) && j ≤ size(P, 2)
        @inbounds begin
            P[i, j] = 0.0
            ε[1][i, j] = 0.0
            ε[2][i, j] = 0.0
            τ[1][i, j] = 0.0
            τ[2][i, j] = 0.0
            R[3][i, j] = λP[i, j]
        end
    end
    if i ≤ size(V[1], 1) && j ≤ size(V[1], 2)
        @inbounds V[1][i, j] = 0.0
    end
    if i ≤ size(V[2], 1) && j ≤ size(V[2], 2)
        @inbounds V[2][i, j] = 0.0
    end
    if i ≤ size(ε[3], 1) && j ≤ size(ε[3], 2)
        @inbounds begin
            ε[3][i, j] = 0.0
            τ[3][i, j] = 0.0
        end
    end
    if i ≤ size(R[1], 1) && j ≤ size(R[1], 2)
        @inbounds R[1][i, j] = λV[1][i + 1, j + 1]
    end
    if i ≤ size(R[2], 1) && j ≤ size(R[2], 2)
        @inbounds R[2][i, j] = λV[2][i + 1, j + 1]
    end
    return nothing
end

# Apply the adjoint Schur-complement correction and diagonal preconditioner,
# then advance the damped velocity iterate. Keep the unpreconditioned corrected
# residual in Rx/Ry so the caller can evaluate the same norm as before fusion.
@parallel_indices (i, j) function update_adjoint_V_damping_DR!(
        Vx,
        Vy,
        dVxdτ,
        dVydτ,
        Rx,
        Ry,
        P,
        γ_eff,
        Dx,
        Dy,
        αVx,
        αVy,
        βVx,
        βVy,
        dτVx,
        dτVy,
        _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            iE = wrap_next(i, size(γ_eff, 1))
            Rx_ij = Rx[i + 1, j + 1] -
                (γ_eff[i, j] * P[i, j] - γ_eff[iE, j] * P[iE, j]) * _di_center[1]
            Rx[i + 1, j + 1] = Rx_ij
            dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
            dVxdτ[i, j] = dVx_new
            Vx[i + 1, j + 1] += ΔVx
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            jN = wrap_next(j, size(γ_eff, 2))
            Ry_ij = Ry[i + 1, j + 1] -
                (γ_eff[i, j] * P[i, j] - γ_eff[i, jN] * P[i, jN]) * _di_center[2]
            Ry[i + 1, j + 1] = Ry_ij
            dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij / Dy[i, j], αVy[i, j], βVy[i, j], dτVy[i, j])
            dVydτ[i, j] = dVy_new
            Vy[i + 1, j + 1] += ΔVy
        end
    end
    return nothing
end
