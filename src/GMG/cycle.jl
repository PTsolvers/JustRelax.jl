# Block residual norms over owned retained DOFs, reduced on the level communicator.
function gmg_block_norms(op::JustRelax.GMGLevelOperator{N}, level, r::JustRelax.GMGCoupledFields{N}) where {N}
    velocity = sum(C -> sum(op.velocity_mask[C] .* abs2.(r.v[C])), 1:N)
    pressure = sum(op.pressure_mask .* abs2.(r.p))
    return sqrt(JustRelax.level_sum(level, velocity)), sqrt(JustRelax.level_sum(level, pressure))
end

gmg_zero!(fields::JustRelax.GMGCoupledFields) = foreach(x -> fill!(x, zero(eltype(x))), (fields.p, fields.v...))

# r = rhs - K e on owned rows (K e must be in op.out); halos are exchanged.
function gmg_residual!(op::JustRelax.GMGLevelOperator{N}, r, rhs, e) where {N}
    JustRelax.gmg_prepare!(op, e)
    gmg_apply_operator!(op, e)
    for C in 1:N
        r.v[C] .= ifelse.(iszero.(op.velocity_mask[C]), zero(eltype(r.p)), rhs.v[C] .- op.out.v[C])
    end
    r.p .= ifelse.(iszero.(op.pressure_mask), zero(eltype(r.p)), rhs.p .- op.out.p)
    JustRelax.exchange_level!(op.exchange, r)
    return nothing
end

# Fixed block scales: a zero block borrows the other block's scale.
@inline function gmg_scales(rv, rp)
    T = typeof(rv)
    sv = rv > 0 ? rv : (rp > 0 ? rp : one(T))
    sp = rp > 0 ? rp : sv
    return sv, sp
end

@inline gmg_combined(rv, rp, (sv, sp)) = sqrt((rv / sv)^2 + (rp / sp)^2)

@inline gmg_sweep_cost(smoother) = smoother.substeps + 1

# Fine-grid-equivalent cost of one operator application on level l (local cells).
gmg_weight(mg, l) = length(mg.hierarchy.levels[l].correction.p) / length(first(mg.hierarchy.levels).correction.p)

@inline function gmg_count!(counters::JustRelax.GMGCounters, n, weight)
    counters.applications += n
    counters.work += n * weight
    return nothing
end

"""
    gmg_coarse_solve!(mg, e, rhs, counters)

Relax the coarsest level from `e = 0` until the block-normalized residual drops
by `mg.coarse.drop`. Returns the status; the stopping test is collective.
"""
function gmg_coarse_solve!(mg::JustRelax.GMGMultigrid, e, rhs, counters::JustRelax.GMGCounters)
    level, op = last(mg.hierarchy.levels), last(mg.operators)
    solver = mg.coarse
    gmg_zero!(e)
    JustRelax.gmg_prepare!(op, e)
    norms = gmg_block_norms(op, level, rhs)
    scales = gmg_scales(norms...)
    initial = gmg_combined(norms..., scales)
    iszero(initial) && return JustRelax.gmg_converged
    for _ in 1:solver.max_sweeps
        gmg_dr_smooth!(op, e, rhs, solver.smoother)
        gmg_residual!(op, level.residual, rhs, e)
        counters.coarse_sweeps += 1
        gmg_count!(counters, gmg_sweep_cost(solver.smoother) + 1, gmg_weight(mg, length(mg.hierarchy.levels)))
        current = gmg_combined(gmg_block_norms(op, level, level.residual)..., scales)
        isfinite(current) || return JustRelax.gmg_nonfinite
        current <= solver.drop * initial && return JustRelax.gmg_converged
    end
    return JustRelax.gmg_budget_exhausted
end

"""
    gmg_vcycle!(mg, l, e, rhs, counters)

One V-cycle for `K_l e = rhs` on level `l`, improving `e` in place. Coarse
corrections start from zero; recurrences reset after every coarse correction.
"""
function gmg_vcycle!(mg::JustRelax.GMGMultigrid, l::Int, e, rhs, counters::JustRelax.GMGCounters)
    levels, ops = mg.hierarchy.levels, mg.operators
    if l == length(levels)
        counters.coarse_status = gmg_coarse_solve!(mg, e, rhs, counters)
        return nothing
    end
    level, op = levels[l], ops[l]
    l > 1 && gmg_zero!(e)
    for _ in 1:mg.pre
        gmg_dr_smooth!(op, e, rhs, mg.smoother)
        counters.sweeps += 1
        gmg_count!(counters, gmg_sweep_cost(mg.smoother), gmg_weight(mg, l))
    end
    gmg_residual!(op, level.residual, rhs, e)
    gmg_count!(counters, 1, gmg_weight(mg, l))
    coarse = levels[l + 1]
    gmg_restrict!(coarse.rhs, level.residual, mg.transfers[l])
    gmg_vcycle!(mg, l + 1, coarse.correction, coarse.rhs, counters)
    # A nonfinite coarse correction is never applied.
    counters.coarse_status == JustRelax.gmg_nonfinite && return nothing
    JustRelax.gmg_prepare!(ops[l + 1], coarse.correction)
    gmg_prolong!(e, coarse.correction, mg.transfers[l])
    JustRelax.gmg_prepare!(op, e)
    for _ in 1:mg.post
        gmg_dr_smooth!(op, e, rhs, mg.smoother)
        counters.sweeps += 1
        gmg_count!(counters, gmg_sweep_cost(mg.smoother), gmg_weight(mg, l))
    end
    return nothing
end

"""
    gmg_refresh!(mg)

Restrict the finest coefficients through the hierarchy and refresh every
level's diagonal and spectral bound (a new operator epoch).
"""
function gmg_refresh!(mg::JustRelax.GMGMultigrid)
    levels, ops = mg.hierarchy.levels, mg.operators
    for l in 1:(length(levels) - 1)
        gmg_restrict_coefficients!(ops[l + 1], ops[l], mg.transfers[l])
    end
    for (level, op) in zip(levels, ops)
        gmg_refresh_bounds!(op, level.partition)
    end
    return nothing
end

"""
    gmg_solve!(mg, e, rhs; rtol = 1e-8, max_cycles = 100, stagnation = 0.99)

Linear V-cycle iteration for `K e = rhs` on the finest level. Stops on a
block-normalized residual drop of `rtol`, a cycle budget, a cycle whose
contraction exceeds `stagnation`, or a nonfinite residual.
Returns `(status, counters, history)` with the normalized residual per cycle.
"""
function gmg_solve!(mg::JustRelax.GMGMultigrid, e, rhs; rtol = 1.0e-8, max_cycles::Integer = 100, stagnation = 0.99)
    level, op = first(mg.hierarchy.levels), first(mg.operators)
    counters = JustRelax.GMGCounters()
    T = eltype(e.p)
    history = T[]
    gmg_residual!(op, level.residual, rhs, e)
    norms = gmg_block_norms(op, level, level.residual)
    scales = gmg_scales(norms...)
    previous = gmg_combined(norms..., scales)
    push!(history, previous)
    iszero(previous) && return JustRelax.gmg_converged, counters, history
    initial = previous
    for _ in 1:max_cycles
        gmg_vcycle!(mg, 1, e, rhs, counters)
        counters.cycles += 1
        counters.coarse_status == JustRelax.gmg_nonfinite && return JustRelax.gmg_nonfinite, counters, history
        gmg_residual!(op, level.residual, rhs, e)
        gmg_count!(counters, 1, 1)
        current = gmg_combined(gmg_block_norms(op, level, level.residual)..., scales)
        push!(history, current)
        isfinite(current) || return JustRelax.gmg_nonfinite, counters, history
        current <= rtol * initial && return JustRelax.gmg_converged, counters, history
        # The first cycle may grow a block transiently (rough pressure RHS); judge contraction afterwards.
        counters.cycles > 1 && current > stagnation * previous && return JustRelax.gmg_stagnated, counters, history
        previous = current
    end
    return JustRelax.gmg_budget_exhausted, counters, history
end
