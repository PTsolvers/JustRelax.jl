# Shared initialization and velocity-update utilities for the adjoint solvers.
function initialize_adjoint_iteration!(adjoint, ni)
    @parallel (@idx ni .+ 2) _initialize_adjoint_iteration!(
        adjoint.P,
        (adjoint.V.Vx, adjoint.V.Vy),
        (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c, adjoint.τ.II),
        (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        (adjoint.R.Rx, adjoint.R.Ry, adjoint.R.RP),
        (adjoint.λV.Vx, adjoint.λV.Vy),
        adjoint.λP,
    )
    return nothing
end

@inline function set_adjoint_dyrel_parameters!(
        dτV, βV, αV, cV, λmaxV, amplification, CFL, i, j,
    )
    dτV_ij = dτV[i, j] = 2 / sqrt(λmaxV[i, j] * amplification) * CFL
    cdt = cV[i, j] * dτV_ij
    inv_den = inv(2 + cdt)
    βV[i, j] = 2 * dτV_ij * inv_den
    αV[i, j] = (2 - cdt) * inv_den
    return nothing
end

# Build the adjoint acceleration parameters from the local implicit constitutive tangent. Each
# velocity row uses the largest amplification among the same center and vertex stress locations
# sampled by its forward Gershgorin row.
function update_adjoint_dτV_α_β!(dyrel, amplification_center, amplification_vertex)
    @parallel (@idx size(amplification_center)) _update_adjoint_dτV_α_β!(
        dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy,
        dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy,
        amplification_center, amplification_vertex, dyrel.CFL,
    )
    return nothing
end

@parallel_indices (i, j) function _update_adjoint_dτV_α_β!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy,
        amplification_center, amplification_vertex, CFL,
    )
    @inbounds begin
        if i ≤ size(dτVx, 1) && j ≤ size(dτVx, 2)
            iE = wrap_next(i, size(amplification_center, 1))
            amplification = max(
                amplification_center[i, j], amplification_center[iE, j],
                amplification_vertex[i + 1, j], amplification_vertex[i + 1, j + 1],
            )
            set_adjoint_dyrel_parameters!(
                dτVx, βVx, αVx, cVx, λmaxVx, amplification, CFL, i, j,
            )
        end
        if i ≤ size(dτVy, 1) && j ≤ size(dτVy, 2)
            jN = wrap_next(j, size(amplification_center, 2))
            amplification = max(
                amplification_center[i, j], amplification_center[i, jN],
                amplification_vertex[i, j + 1], amplification_vertex[i + 1, j + 1],
            )
            set_adjoint_dyrel_parameters!(
                dτVy, βVy, αVy, cVy, λmaxVy, amplification, CFL, i, j,
            )
        end
    end
    return nothing
end

# The variational stress kernel obtains vertex viscosity by harmonic interpolation from centers.
# Use every center contributing to the two vertex stresses in a momentum row when choosing its
# local tangent amplification.
function update_variational_adjoint_dτV_α_β!(dyrel, amplification_center)
    @parallel (@idx size(amplification_center)) _update_variational_adjoint_dτV_α_β!(
        dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy,
        dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy,
        amplification_center, dyrel.CFL,
    )
    return nothing
end

@parallel_indices (i, j) function _update_variational_adjoint_dτV_α_β!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy,
        amplification_center, CFL,
    )
    @inbounds begin
        nx, ny = size(amplification_center)
        periodic_x = size(dτVx, 1) == nx
        periodic_y = size(dτVy, 2) == ny
        iW = i == 1 ? (periodic_x ? nx : 1) : i - 1
        iE = i == nx ? (periodic_x ? 1 : nx) : i + 1
        jS = j == 1 ? (periodic_y ? ny : 1) : j - 1
        jN = j == ny ? (periodic_y ? 1 : ny) : j + 1

        if i ≤ size(dτVx, 1) && j ≤ size(dτVx, 2)
            amplification = max(
                amplification_center[i, jS], amplification_center[iE, jS],
                amplification_center[i, j], amplification_center[iE, j],
                amplification_center[i, jN], amplification_center[iE, jN],
            )
            set_adjoint_dyrel_parameters!(
                dτVx, βVx, αVx, cVx, λmaxVx, amplification, CFL, i, j,
            )
        end
        if i ≤ size(dτVy, 1) && j ≤ size(dτVy, 2)
            amplification = max(
                amplification_center[iW, j], amplification_center[i, j],
                amplification_center[iE, j], amplification_center[iW, jN],
                amplification_center[i, jN], amplification_center[iE, jN],
            )
            set_adjoint_dyrel_parameters!(
                dτVy, βVy, αVy, cVy, λmaxVy, amplification, CFL, i, j,
            )
        end
    end
    return nothing
end

@parallel_indices (i, j) function _initialize_adjoint_iteration!(
        P, V, ε, τ_center, τ_vertex, R, λV, λP
    )
    if i ≤ size(P, 1) && j ≤ size(P, 2)
        @inbounds begin
            P[i, j] = 0.0
            ε[1][i, j] = 0.0
            ε[2][i, j] = 0.0
            τ_center[1][i, j] = 0.0
            τ_center[2][i, j] = 0.0
            τ_center[3][i, j] = 0.0
            τ_center[4][i, j] = 0.0
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
            τ_vertex[1][i, j] = 0.0
            τ_vertex[2][i, j] = 0.0
            τ_vertex[3][i, j] = 0.0
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

# Masked counterpart of `update_adjoint_V_damping_DR!`. The Schur-complement correction is the
# transpose of the forward penalty, whose continuity residual carries the rock fraction; rows
# eliminated by `ϕ` are held at zero, as in the forward `compute_DR_residual_update_V!`.
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
        ϕ::JustRelax.RockRatio,
        _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            if isvalid_vx(ϕ, i + 1, j)
                iE = wrap_next(i, size(γ_eff, 1))
                Rx_ij = Rx[i + 1, j + 1] - (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[iE, j] * γ_eff[iE, j] * P[iE, j]
                ) * @dx(_di_center, i)
                Rx[i + 1, j + 1] = Rx_ij
                dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
                dVxdτ[i, j] = dVx_new
                Vx[i + 1, j + 1] += ΔVx
            else
                Rx[i + 1, j + 1] = 0.0
                dVxdτ[i, j] = 0.0
                Vx[i + 1, j + 1] = 0.0
            end
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            if isvalid_vy(ϕ, i, j + 1)
                jN = wrap_next(j, size(γ_eff, 2))
                Ry_ij = Ry[i + 1, j + 1] - (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[i, jN] * γ_eff[i, jN] * P[i, jN]
                ) * @dy(_di_center, j)
                Ry[i + 1, j + 1] = Ry_ij
                dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij / Dy[i, j], αVy[i, j], βVy[i, j], dτVy[i, j])
                dVydτ[i, j] = dVy_new
                Vy[i + 1, j + 1] += ΔVy
            else
                Ry[i + 1, j + 1] = 0.0
                dVydτ[i, j] = 0.0
                Vy[i + 1, j + 1] = 0.0
            end
        end
    end
    return nothing
end
