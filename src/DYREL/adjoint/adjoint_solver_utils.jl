# Setup and residual utilities of the adjoint solve. The inner velocity solvers (damped dynamic
# relaxation and BiCGStab) are in `adjoint_inner_solvers.jl`.
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
# `adjoint_kwargs`.
function adjoint_solver_options(forward::NamedTuple, adjoint_kwargs)
    adjoint_kwargs isa NamedTuple ||
        throw(ArgumentError("`adjoint_kwargs` must be a NamedTuple, got $(typeof(adjoint_kwargs))"))
    unknown = filter(∉(ADJOINT_SOLVER_OPTIONS), keys(adjoint_kwargs))
    isempty(unknown) || throw(
        ArgumentError("unknown adjoint solver option(s) $unknown; supported: $ADJOINT_SOLVER_OPTIONS")
    )
    iterMax_DR = get(adjoint_kwargs, :iterMax_DR, get(adjoint_kwargs, :iterMax, forward.iterMax_DR))
    return merge(forward, Base.structdiff(adjoint_kwargs, NamedTuple{(:iterMax,)}), (; iterMax_DR))
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

@parallel_indices (i, j) function local_adjoint_penalty!(Γ, η, ηb, ϕ, factor)
    @inbounds begin
        weight = isvalid_c(ϕ, i, j) ? ϕ.center[i, j] : 0.0
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
    Gershgorin_Stokes2D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy,
        stokes.viscosity.η, dyrel.γ_eff, phases, rock, rheology, grid.di, dt,
        free_surface ? ρg[end] : nothing,
    )
    update_dτV_α_β!(dyrel)
    return nothing
end

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
