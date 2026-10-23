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

# The adjoint iterates on the consistent tangent of the converged forward state, whose smallest
# eigenvalue can lie far from the one the forward damping was tuned to (a yielding plastic
# tangent is much softer). Re-estimate the damping every `nout` iterations as the forward
# solver does: λmin is the Rayleigh quotient of the last velocity step against the change of the
# preconditioned residual it caused. The residual of a step only exists in the next iteration,
# so the step and its residual are stored one iteration before each estimate.
struct AdjointDamping{C, M}
    cV0::C         # forward damping, restored after the adjoint solve
    masks::M       # momentum rows entering the estimate
    stored::Base.RefValue{Int}  # iteration of the stored step, 0 if none
end

function AdjointDamping(dyrel, masks)
    return AdjointDamping((copy(dyrel.cVx), copy(dyrel.cVy)), masks, Ref(0))
end

AdjointDamping(dyrel) = AdjointDamping(
    dyrel, (fill!(similar(dyrel.Dx, Bool), true), fill!(similar(dyrel.Dy, Bool), true))
)

@inline preconditioned_residual(R, D) = iszero(D) ? zero(R) : R / D

"""
    update_adjoint_damping!(damping, dyrel, residuals, iter, nout, ni, update_steps!)

Store the last velocity step before an iteration that is a multiple of `nout`; at that
iteration, update the damping `c = 2√λmin·c_fact` from the stored step and call
`update_steps!()` to recompute the pseudo-time steps. `residuals` are the unpreconditioned
adjoint momentum residuals of the current iterate.
"""
function update_adjoint_damping!(damping, dyrel, residuals, iter, nout, ni, update_steps!)
    if iszero((iter + 1) % nout)
        @parallel (@idx ni) compute_dV!(
            (dyrel.dVx, dyrel.dVy), (dyrel.dVxdτ, dyrel.dVydτ),
            (dyrel.βVx, dyrel.βVy), (dyrel.dτVx, dyrel.dτVy),
        )
        @. dyrel.Rx0 = preconditioned_residual(residuals[1], dyrel.Dx)
        @. dyrel.Ry0 = preconditioned_residual(residuals[2], dyrel.Dy)
        damping.stored[] = iter
    elseif iszero(iter % nout) && damping.stored[] == iter - 1
        # a stored step from before a λP update would mix in the pressure change
        λminV = adjoint_λminV(
            (dyrel.dVx, dyrel.dVy), residuals, (dyrel.Dx, dyrel.Dy), (dyrel.Rx0, dyrel.Ry0),
            damping.masks,
        )
        @parallel (@idx ni) update_cV!((dyrel.cVx, dyrel.cVy), 2 * √(λminV) * dyrel.c_fact)
        update_steps!()
    end
    return nothing
end

function adjoint_λminV(dV, residuals, D, residuals0, masks)
    numerator = sum(
        ntuple(Val(2)) do d
            sum_mpi(
                (m, dv, r, Dd, r0) -> m ? dv * (preconditioned_residual(r, Dd) - r0) : zero(dv),
                masks[d], dV[d], residuals[d], D[d], residuals0[d],
            )
        end
    )
    denominator = sum(
        ntuple(d -> sum_mpi((m, dv) -> m ? abs2(dv) : zero(abs2(dv)), masks[d], dV[d]), Val(2))
    )
    return rayleigh_quotient(numerator, denominator)
end

# Hand the shared DYREL state back to the forward solver: its damping, the pseudo-time steps
# derived from it, and no velocity history.
function restore_forward_damping!(damping, dyrel)
    copyto!(dyrel.cVx, damping.cV0[1])
    copyto!(dyrel.cVy, damping.cV0[2])
    update_dτV_α_β!(dyrel)
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0
    return nothing
end

# A diverged adjoint solve leaves no usable starting point for the next one.
function reset_nonfinite_adjoint!(adjoint)
    λ = (adjoint.λV.Vx, adjoint.λV.Vy, adjoint.λP)
    all(A -> all(isfinite, A), λ) || foreach(A -> fill!(A, 0), λ)
    return nothing
end

# A pressure objective seeds only the continuity adjoint, so its momentum residual at λ = 0 is
# zero. Measure it instead against the momentum residual a cold start reaches after its first
# outer pass: λV = 0 and λP from the pressure seed, which `stokes_ad.P` holds on entry.
# `evaluate_residual!()` returns the momentum residual norms of the current λ.
function pressure_objective_reference!(evaluate_residual!, stokes_ad, dyrel, pressure_relaxation)
    λ = (stokes_ad.λV.Vx, stokes_ad.λV.Vy, stokes_ad.λP)
    saved = map(copy, λ)
    fill!(stokes_ad.λV.Vx, 0)
    fill!(stokes_ad.λV.Vy, 0)
    @. stokes_ad.λP = pressure_relaxation * dyrel.γ_eff * stokes_ad.P
    errV = evaluate_residual!()
    foreach(copyto!, λ, saved)
    return errV
end

# Reference of the relative momentum residual: the first nonzero one. A pressure objective
# seeds only the continuity adjoint, so the momentum residual stays zero until λP is nonzero.
function momentum_residual_reference(errV0, errV)
    iszero(maximum(errV0)) || return errV0
    errV_max = maximum(errV)
    return map(_ -> errV_max, errV)
end

relative_momentum_residual(errV, errV0) =
    map((e, e0) -> iszero(e0) ? zero(e) : e / e0, errV, errV0)

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
                (γ_eff[i, j] * P[i, j] - γ_eff[iE, j] * P[iE, j]) * @dx(_di_center, i)
            Rx[i + 1, j + 1] = Rx_ij
            dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
            dVxdτ[i, j] = dVx_new
            Vx[i + 1, j + 1] += ΔVx
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            jN = wrap_next(j, size(γ_eff, 2))
            Ry_ij = Ry[i + 1, j + 1] -
                (γ_eff[i, j] * P[i, j] - γ_eff[i, jN] * P[i, jN]) * @dy(_di_center, j)
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
