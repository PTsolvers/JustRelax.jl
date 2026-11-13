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

const ADJOINT_SOLVER_OPTIONS = (
    :ϵ, :rel_drop, :iterMax, :iterMax_PH, :iterMax_DR, :total_iterMax, :nout,
    :verbose_PH, :verbose_DR, :pressure_relaxation, :penalty_scaling, :penalty_factor,
    :inner_solver, :deflation,
)

# Iteration controls of the adjoint solve: those of the forward solve, overridden by
# `adjoint_kwargs`. `kwargs` are the remaining keywords of the forward call.
function adjoint_solver_options(forward::NamedTuple, adjoint_kwargs, kwargs)
    adjoint_kwargs isa NamedTuple ||
        throw(ArgumentError("`adjoint_kwargs` must be a NamedTuple, got $(typeof(adjoint_kwargs))"))
    unknown = filter(∉(ADJOINT_SOLVER_OPTIONS), keys(adjoint_kwargs))
    isempty(unknown) || throw(
        ArgumentError("unknown adjoint solver option(s) $unknown; supported: $ADJOINT_SOLVER_OPTIONS")
    )
    options = (; kwargs...)
    for (name, alias) in ((:ϵ, :adjoint_ϵ), (:rel_drop, :adjoint_rel_drop))
        haskey(adjoint_kwargs, name) && !isnothing(get(options, alias, nothing)) &&
            throw(ArgumentError("give either `adjoint_kwargs.$name` or `$alias`, not both"))
    end
    iterMax_DR = get(adjoint_kwargs, :iterMax_DR, get(adjoint_kwargs, :iterMax, forward.iterMax_DR))
    ϵ = haskey(adjoint_kwargs, :ϵ) ? (; adjoint_ϵ = adjoint_kwargs.ϵ) : (;)
    overrides = Base.structdiff(adjoint_kwargs, NamedTuple{(:ϵ, :iterMax)})
    return merge(forward, options, overrides, (; iterMax_DR), ϵ)
end

# The local numerical penalty leaves the physical bulk viscosity and the constitutive tangent
# unchanged. `:global` inherits the forward operator exactly.
function adjoint_penalty_factor(scaling, factor)
    scaling in (:global, :local) ||
        throw(ArgumentError("penalty_scaling must be :global or :local"))
    if scaling === :global
        isnothing(factor) || throw(ArgumentError("penalty_factor requires penalty_scaling = :local"))
        return nothing
    end
    factor = something(factor, 320.0)
    factor isa Real && isfinite(factor) && factor > 0 ||
        throw(ArgumentError("penalty_factor must be finite and positive"))
    return factor
end

@inline adjoint_penalty_weight(::Nothing, i, j) = 1.0
@inline adjoint_penalty_weight(ϕ::JustRelax.RockRatio, i, j) =
    isvalid_c(ϕ, i, j) ? ϕ.center[i, j] : 0.0

@parallel_indices (i, j) function local_adjoint_penalty!(Γ, η, ηb, ϕ, factor)
    @inbounds begin
        weight = adjoint_penalty_weight(ϕ, i, j)
        if iszero(weight)
            Γ[i, j] = 0.0
        else
            numerical = factor * η[i, j]
            physical = isinf(ηb[i, j]) ? numerical : ηb[i, j]
            # Harmonic combination, without multiplying dimensional viscosities together.
            lo, hi = minmax(physical, numerical)
            Γ[i, j] = lo / (1 + lo / hi) / weight
        end
    end
    return nothing
end

function configure_adjoint_penalty!(dyrel, stokes, phases, rock, rheology, grid, dt, ρg, free_surface, factor)
    isnothing(factor) && return nothing
    @parallel (@idx size(stokes.P)) local_adjoint_penalty!(
        dyrel.γ_eff, stokes.viscosity.η, dyrel.ηb, rock, factor,
    )
    sum_mpi(x -> !(isfinite(x) && x ≥ 0), dyrel.γ_eff) == 0 ||
        throw(ArgumentError("local adjoint penalty must be finite and nonnegative"))
    if isnothing(rock)
        Gershgorin_Stokes2D_SchurComplement!(
            dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
            stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phases, rheology, grid.di, dt,
        )
        if free_surface
            apply_free_surface_diagonal!(dyrel.Dy, dyrel.λmaxVy, ρg[end], grid.di.center, dt)
        end
    else
        Gershgorin_Stokes2D_SchurComplement!(
            dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
            stokes.viscosity.η, dyrel.γ_eff, phases, rock, rheology, grid.di, dt,
            free_surface ? ρg[end] : nothing,
        )
    end
    update_dτV_α_β!(dyrel)
    return nothing
end

# The adjoint iterates on the consistent tangent of the converged forward state, whose smallest
# eigenvalue can lie far from the one the forward damping was tuned to (a yielding plastic
# tangent is much softer). Re-estimate the damping every `nout` iterations as the forward
# solver does: λmin is the Rayleigh quotient of the last velocity step against the change of the
# preconditioned residual it caused. The residual of a step only exists in the next iteration,
# so the step and its residual are stored one iteration before each estimate.
struct AdjointDamping{M}
    masks::M       # momentum rows entering the estimate
    stored::Base.RefValue{Int}  # iteration of the stored step, 0 if none
end

AdjointDamping(masks) = AdjointDamping(masks, Ref(0))

AdjointDamping(dyrel::JustRelax.DYREL) = AdjointDamping(
    (fill!(similar(dyrel.Dx, Bool), true), fill!(similar(dyrel.Dy, Bool), true))
)

"""
    adjoint_dyrel!(stokes_ad, dyrel)

Return the adjoint's own `DYREL` state, kept in `stokes_ad` between adjoint solves, so that the
forward `dyrel` is never modified. The operator — penalty, bulk viscosity, preconditioner and
eigenvalue bounds — and the scalar parameters are those of the forward `dyrel` at the current
state. The damping carries over from the previous adjoint solve; the first solve starts from the
forward damping. The pseudo-velocity history starts at zero.
"""
function adjoint_dyrel!(stokes_ad, dyrel::JustRelax.DYREL)
    names = fieldnames(JustRelax.DYREL)
    previous = stokes_ad.solver.dyrel
    reuse = typeof(previous) === typeof(dyrel) && all(names) do name
        A = getfield(dyrel, name)
        !(A isa AbstractArray) || size(A) == size(getfield(previous, name))
    end
    fields = map(names) do name
        A = getfield(dyrel, name)
        A isa AbstractArray || return A
        reuse || return copy(A)
        B = getfield(previous, name)
        name in (:cVx, :cVy, :cVz) || copyto!(B, A)
        return B
    end
    adjoint = JustRelax.DYREL(fields...)
    fill!(adjoint.dVxdτ, 0)
    fill!(adjoint.dVydτ, 0)
    stokes_ad.solver.dyrel = adjoint
    return adjoint
end

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

masked_dot_mpi(::Nothing, A, B) = sum_mpi((a, b) -> a * b, A, B)
masked_dot_mpi(mask, A, B) = sum_mpi((m, a, b) -> m ? a * b : zero(a * b), mask, A, B)

# A pressure objective seeds only the continuity adjoint, so its momentum residual at λ = 0 is
# zero. Measure it instead against the momentum residual after the first outer pass: λV = 0 and
# λP from the pressure seed, which `stokes_ad.P` holds on entry. Expects λ = 0 and leaves it so.
# `evaluate_residual!()` returns the momentum residual norms of the current λ.
function pressure_objective_reference!(evaluate_residual!, stokes_ad, dyrel, pressure_relaxation)
    @. stokes_ad.λP = pressure_relaxation * dyrel.γ_eff * stokes_ad.P
    errV = evaluate_residual!()
    fill!(stokes_ad.λP, 0)
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

# Per-iteration reset. Enzyme zeroes the shadow of every output it propagates, so one reverse
# sweep consumes the stress, strain-rate and residual seeds; only the accumulated residuals P and
# V need zeroing. `initialize_adjoint_iteration!` clears everything, once per solve.
# `λV` and `λP` are the point the residual is evaluated at: the adjoint unknowns, or a search
# direction of the Krylov inner solver.
function reset_adjoint_residuals!(
        adjoint, ni, λV = (adjoint.λV.Vx, adjoint.λV.Vy), λP = adjoint.λP,
    )
    @parallel (@idx ni .+ 2) _reset_adjoint_residuals!(
        adjoint.P,
        (adjoint.V.Vx, adjoint.V.Vy),
        (adjoint.R.Rx, adjoint.R.Ry, adjoint.R.RP),
        λV,
        λP,
    )
    return nothing
end

@parallel_indices (i, j) function _reset_adjoint_residuals!(P, V, R, λV, λP)
    if i ≤ size(P, 1) && j ≤ size(P, 2)
        @inbounds begin
            P[i, j] = 0.0
            R[3][i, j] = λP[i, j]
        end
    end
    if i ≤ size(V[1], 1) && j ≤ size(V[1], 2)
        @inbounds V[1][i, j] = 0.0
    end
    if i ≤ size(V[2], 1) && j ≤ size(V[2], 2)
        @inbounds V[2][i, j] = 0.0
    end
    if i ≤ size(R[1], 1) && j ≤ size(R[1], 2)
        @inbounds R[1][i, j] = λV[1][i + 1, j + 1]
    end
    if i ≤ size(R[2], 1) && j ≤ size(R[2], 2)
        @inbounds R[2][i, j] = λV[2][i + 1, j + 1]
    end
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

# The Schur-complement correction of `update_adjoint_V_damping_DR!` on its own: the Krylov inner
# solver needs the corrected residual without the damped update.
@parallel_indices (i, j) function adjoint_schur_correction!(Rx, Ry, P, γ_eff, Dx, Dy, _di_center)
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            iE = wrap_next(i, size(γ_eff, 1))
            Rx[i + 1, j + 1] -=
                (γ_eff[i, j] * P[i, j] - γ_eff[iE, j] * P[iE, j]) * @dx(_di_center, i)
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            jN = wrap_next(j, size(γ_eff, 2))
            Ry[i + 1, j + 1] -=
                (γ_eff[i, j] * P[i, j] - γ_eff[i, jN] * P[i, jN]) * @dy(_di_center, j)
        end
    end
    return nothing
end

@parallel_indices (i, j) function adjoint_schur_correction!(
        Rx, Ry, P, γ_eff, Dx, Dy, ϕ::JustRelax.RockRatio, _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            if isvalid_vx(ϕ, i + 1, j)
                iE = wrap_next(i, size(γ_eff, 1))
                Rx[i + 1, j + 1] -= (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[iE, j] * γ_eff[iE, j] * P[iE, j]
                ) * @dx(_di_center, i)
            else
                Rx[i + 1, j + 1] = 0.0
            end
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            if isvalid_vy(ϕ, i, j + 1)
                jN = wrap_next(j, size(γ_eff, 2))
                Ry[i + 1, j + 1] -= (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[i, jN] * γ_eff[i, jN] * P[i, jN]
                ) * @dy(_di_center, j)
            else
                Ry[i + 1, j + 1] = 0.0
            end
        end
    end
    return nothing
end

function check_inner_solver(inner_solver)
    inner_solver in (:dr, :bicgstab) ||
        throw(ArgumentError("inner_solver must be :dr or :bicgstab"))
    return inner_solver
end

function check_deflation(deflation, inner_solver)
    deflation isa Integer && deflation ≥ 0 || throw(
        ArgumentError("deflation must be a nonnegative integer, the number of recycled vectors")
    )
    iszero(deflation) || inner_solver === :bicgstab ||
        throw(ArgumentError("deflation requires inner_solver = :bicgstab"))
    return deflation
end

adjoint_dot(a, b) = sum(map((a, b) -> masked_dot_mpi(nothing, a, b), a, b))

# Recycled deflation of the Krylov inner solver. A few very slow modes of the velocity system,
# e.g. the rigid motions of a stiff body enclosed by a weak layer, cost the diagonally
# preconditioned iteration most of its sweeps. What an inexact solve leaves behind is dominated
# by those modes, so the solution increments of the pressure passes span them: BiCGStab then
# iterates on P A with P = I − AW E Wᵀ and E = (Wᵀ A W)⁻¹, where W holds the last `kmax`
# increments (orthonormalized) of the scaled system.
mutable struct AdjointDeflation{A, V}
    const W::A              # per velocity component, the vectors in the slices of one array
    const AW::A             # their images: differences of true residuals, so they cost no sweep
    E::Matrix{Float64}      # (Wᵀ A W)⁻¹ on the filled slots, zero elsewhere
    const slots::Vector{Int}    # filled slots, oldest first
    const x0::V             # iterate and scaled residual at the start of the pass
    const r0::V
    const scratch::V
end

function AdjointDeflation(vectors, kmax)
    slices() = map(A -> fill!(similar(A, size(A)..., kmax), 0), vectors())
    return AdjointDeflation(slices(), slices(), zeros(kmax, kmax), Int[], vectors(), vectors(), vectors())
end

# one column per vector, so that all vectors are handled by a single matrix product
@inline columns(A) = reshape(A, :, size(A, ndims(A)))

# Wᵀ y for all slots at once, summed over the velocity components and the MPI ranks
function deflation_dots(W, y)
    c = sum(map((W, y) -> Array(columns(W)' * vec(y)), W, y))
    return MPI.Allreduce(c, MPI.SUM, MPI.COMM_WORLD)
end

on_backend(c, A) = copyto!(similar(A, length(c)), c)

# y ← P y. Returns c = E Wᵀ y, which also gives the iterate correction Q x = x − W c for y = A x.
deflate!(y, ::Nothing) = nothing
function deflate!(y, deflation::AdjointDeflation)
    isempty(deflation.slots) && return nothing
    c = on_backend(deflation.E * deflation_dots(deflation.W, y), first(y))
    for d in eachindex(y)
        mul!(vec(y[d]), columns(deflation.AW[d]), c, -1.0, 1.0)
    end
    return c
end

# λ += s W c, in the unscaled unknowns
correct_iterate!(λVi, deflation, ::Nothing, s, isqrtD) = nothing
function correct_iterate!(λVi, deflation, c, s, isqrtD)
    for d in eachindex(λVi)
        mul!(vec(deflation.scratch[d]), columns(deflation.W[d]), c)
        @. λVi[d] += s * deflation.scratch[d] * isqrtD[d]
    end
    return nothing
end

# Remember where the pass starts and solve for the deflated modes: λ += W E Wᵀ r, r ← P r.
start_deflated_pass!(::Nothing, λVi, r, isqrtD) = nothing
function start_deflated_pass!(deflation, λVi, r, isqrtD)
    for d in eachindex(r)
        deflation.x0[d] .= λVi[d]
        deflation.r0[d] .= r[d]
    end
    correct_iterate!(λVi, deflation, deflate!(r, deflation), 1.0, isqrtD)
    return nothing
end

# Add the increment of the finished pass to the deflation space. `R` is the true residual of
# the new `λVi`, for the same λP as at the start of the pass.
recycle_increment!(::Nothing, λVi, R, isqrtD) = nothing
function recycle_increment!(deflation, λVi, R, isqrtD)
    (; W, AW, slots) = deflation
    kmax = size(deflation.E, 1)
    Δ, AΔ = deflation.x0, deflation.r0
    for d in eachindex(Δ)
        @. Δ[d] = ifelse(isqrtD[d] > 0, (λVi[d] - Δ[d]) / isqrtD[d], 0.0)
        @. AΔ[d] -= R[d] * isqrtD[d]
    end
    scale = sqrt(adjoint_dot(Δ, Δ))
    # orthogonalize against the vectors that stay; the oldest makes room when all slots are full
    kept = length(slots) == kmax ? slots[2:end] : slots
    h = zeros(kmax)
    h[kept] = deflation_dots(W, Δ)[kept]
    h = on_backend(h, first(Δ))
    for d in eachindex(Δ)
        mul!(vec(Δ[d]), columns(W[d]), h, -1.0, 1.0)
        mul!(vec(AΔ[d]), columns(AW[d]), h, -1.0, 1.0)
    end
    nrm = sqrt(adjoint_dot(Δ, Δ))
    # an increment that is already in the space adds nothing
    isfinite(nrm) && nrm > 1.0e-8 * scale || return nothing
    slot = length(slots) == kmax ? popfirst!(slots) : first(setdiff(1:kmax, slots))
    for d in eachindex(Δ)
        selectdim(W[d], ndims(W[d]), slot) .= Δ[d] ./ nrm
        selectdim(AW[d], ndims(AW[d]), slot) .= AΔ[d] ./ nrm
    end
    push!(slots, slot)
    G = sum(map((W, AW) -> Array(columns(W)' * columns(AW)), W, AW))
    G = MPI.Allreduce(G, MPI.SUM, MPI.COMM_WORLD)
    factorization = lu(G[slots, slots]; check = false)
    fill!(deflation.E, 0)
    if issuccess(factorization)
        deflation.E[slots, slots] = inv(factorization)
    else
        empty!(slots)
    end
    return nothing
end

# Work arrays of `adjoint_bicgstab!`, allocated once per adjoint solve on the backend of `D`.
# `deflation` is the number of recycled deflation vectors, 0 for none.
function adjoint_bicgstab_workspace(stokes_ad, D, deflation = 0)
    direction = (zero(stokes_ad.λV.Vx), zero(stokes_ad.λV.Vy))
    direction_i = map(
        (A, D) -> @view(A[2:(size(D, 1) + 1), 2:(size(D, 2) + 1)]), direction, D,
    )
    vectors() = map(zero, D)
    return (;
        direction, direction_i, zeroP = zero(stokes_ad.λP), isqrtD = vectors(),
        r = vectors(), r̂ = vectors(), p = vectors(), v = vectors(), t = vectors(),
        deflation = iszero(deflation) ? nothing : AdjointDeflation(vectors, deflation),
    )
end

"""
    adjoint_bicgstab!(
        residual!, momentum_norms, work, λVi, R, D, errV0, err, ϵ_vel, iter, iterMax,
        total_iterMax; verbose = false, nout = 1,
    )

Inner velocity solve of one Powell–Hestenes pass of the adjoint with BiCGStab, in place of the
damped dynamic relaxation. The adjoint is linear, so the step and the damping need no
eigenvalue estimates, and BiCGStab does not need the Jacobian to be symmetric or to have real
eigenvalues (non-associated plasticity, the tangent of a nonlinear viscosity). It iterates on
the symmetrically scaled system `D^-½ A D^-½`, with `D` the DYREL diagonal, deflated by the
recycled vectors of `work.deflation` if there are any.

`residual!(λV, λP, seed)` writes the Schur-corrected momentum residual of `(λV, λP)` to `R`,
with or without the objective seed; on entry `R` holds that of the current `λVi`. Returns
`(err, iter)`; `iter` counts reverse sweeps, two per BiCGStab iteration. After a breakdown the
solve warns, returns early and restarts from the true residual in the next pass.
"""
function adjoint_bicgstab!(
        residual!, momentum_norms, work, λVi, R, D, errV0, err, ϵ_vel, iter, iterMax,
        total_iterMax; verbose = false, nout = 1,
    )
    (; direction, direction_i, zeroP, isqrtD, r, r̂, p, v, t, deflation) = work
    components = eachindex(r)
    dot = adjoint_dot
    # out = P D^-½ A D^-½ x, by one reverse sweep without the objective seed: R = −A·direction.
    # Returns the deflation coefficients of the sweep, `nothing` without deflation.
    function apply!(out, x)
        for d in components
            @. direction_i[d] = x[d] * isqrtD[d]
        end
        residual!(direction, zeroP, false)
        for d in components
            @. out[d] = -R[d] * isqrtD[d]
        end
        return deflate!(out, deflation)
    end
    for d in components
        # rows without a diagonal are not unknowns
        @. isqrtD[d] = ifelse(D[d] > 0, inv(sqrt(D[d])), 0.0)
        @. r[d] = R[d] * isqrtD[d]
    end
    start_deflated_pass!(deflation, λVi, r, isqrtD)
    for d in components
        copyto!(r̂[d], r[d])
        fill!(p[d], 0)
        fill!(v[d], 0)
    end
    ρ = α = ω = one(eltype(r[1]))
    itPT = 0
    while err > ϵ_vel && itPT < iterMax && iter < total_iterMax
        itPT += 2
        iter += 2
        ρ_new = dot(r̂, r)
        β = (ρ_new / ρ) * (α / ω)
        ρ = ρ_new
        for d in components
            @. p[d] = r[d] + β * (p[d] - ω * v[d])
        end
        c = apply!(v, p)
        α = ρ / dot(r̂, v)
        if !isfinite(α)
            @warn "adjoint BiCGStab broke down (α = $α); restarting from the true residual" maxlog = 5
            break
        end
        for d in components
            @. λVi[d] += α * p[d] * isqrtD[d]
            @. r[d] -= α * v[d]
        end
        correct_iterate!(λVi, deflation, c, -α, isqrtD)
        c = apply!(t, r)
        ω = dot(t, r) / dot(t, t)
        if !isfinite(ω) || iszero(ω)
            @warn "adjoint BiCGStab broke down (ω = $ω); restarting from the true residual" maxlog = 5
            break
        end
        for d in components
            @. λVi[d] += ω * r[d] * isqrtD[d]
            @. r[d] -= ω * t[d]
            # the unscaled residual, for the norm of the stopping criterion
            @. R[d] = r[d] * D[d] * isqrtD[d]
        end
        correct_iterate!(λVi, deflation, c, -ω, isqrtD)
        err = maximum(relative_momentum_residual(momentum_norms(), errV0))
        verbose && iszero(itPT % 2nout) && @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
    end
    return err, iter
end
