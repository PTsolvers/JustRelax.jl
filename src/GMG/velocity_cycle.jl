# Velocity-block multigrid: V-cycles on K_vv = K_v(·, p = 0), the augmented
# viscous block (A + γ grad-div), used inside an outer PH pressure iteration.
# Every correction keeps p = 0, so K_v of a correction is K_vv alone.

# Velocity rows of r = rhs - K e; r.p = 0. Halos are exchanged.
function gmg_velocity_residual!(op::JustRelax.GMGLevelOperator{N}, r, rhs, e) where {N}
    JustRelax.gmg_prepare!(op, e)
    gmg_apply_operator!(op, e)
    for C in 1:N
        r.v[C] .= ifelse.(iszero.(op.velocity_mask[C]), zero(eltype(r.p)), rhs.v[C] .- op.out.v[C])
    end
    fill!(r.p, zero(eltype(r.p)))
    JustRelax.exchange_level!(op.exchange, r)
    return nothing
end

gmg_velocity_norm(op, level, r) = first(gmg_block_norms(op, level, r))

# ⟨c, r⟩ / ⟨c, K_vv c⟩ in the face-measure inner product (K_vv c in op.out);
# zero if the direction carries no energy.
function gmg_energy_step(op::JustRelax.GMGLevelOperator{N}, level, c, r) where {N}
    num = gmg_velocity_dot(op, level, c.v, r.v)
    den = gmg_velocity_dot(op, level, c.v, op.out.v)
    return den > 0 && isfinite(num) ? num / den : zero(num)
end

function gmg_velocity_dot(op::JustRelax.GMGLevelOperator{N}, level, a, b) where {N}
    w = op.velocity_weight
    return JustRelax.level_sum(level, sum(C -> sum(w[C] .* a[C] .* b[C]), 1:N))
end

"""
    gmg_velocity_fcg!(mg, x, b, counters; rtol = 1e-1, max_cycles = 20)

Flexible CG (one-term recurrence, after Notay 2000) for the finest velocity
block `K_vv x = b_v` from `x = 0`, preconditioned by one velocity V-cycle per
iteration, in the face-measure inner product. Overwrites `b` with the final
velocity residual (`b.p` is untouched). `x.p` stays zero. Stops on a velocity
residual drop of `rtol` or after `max_cycles` V-cycles; returns the status.
"""
function gmg_velocity_fcg!(mg::JustRelax.GMGMultigrid, x, b, counters::JustRelax.GMGCounters; rtol = 1.0e-1, max_cycles::Integer = 20)
    level, op = first(mg.hierarchy.levels), first(mg.operators)
    z, s, q = mg.krylov
    N = length(x.v)
    gmg_zero!(x)
    foreach(x -> fill!(x, zero(eltype(x))), (s.v..., q.v...))
    initial = gmg_velocity_norm(op, level, b)
    iszero(initial) && return JustRelax.gmg_converged
    sq = zero(initial)
    for k in 1:max_cycles
        gmg_zero!(z)
        gmg_velocity_vcycle!(mg, 1, z, b, counters)
        counters.cycles += 1
        counters.coarse_status == JustRelax.gmg_nonfinite && return JustRelax.gmg_nonfinite
        # s = z - (z, K s_old)/(s_old, K s_old) s_old keeps the new direction K-orthogonal.
        β = k == 1 ? zero(sq) : -gmg_velocity_dot(op, level, z.v, q.v) / sq
        foreach((sv, zv) -> sv .= zv .+ β .* sv, s.v, z.v)
        JustRelax.gmg_prepare!(op, s)
        gmg_apply_operator!(op, s)
        gmg_count!(counters, 1, 1)
        for C in 1:N
            q.v[C] .= op.velocity_mask[C] .* op.out.v[C]
        end
        sq = gmg_velocity_dot(op, level, s.v, q.v)
        sq > 0 || return JustRelax.gmg_nonfinite
        α = gmg_velocity_dot(op, level, s.v, b.v) / sq
        foreach((xv, sv) -> xv .+= α .* sv, x.v, s.v)
        JustRelax.gmg_prepare!(op, x)
        foreach((bv, qv) -> bv .-= α .* qv, b.v, q.v)
        current = gmg_velocity_norm(op, level, b)
        isfinite(current) || return JustRelax.gmg_nonfinite
        current <= rtol * initial && return JustRelax.gmg_converged
    end
    return JustRelax.gmg_budget_exhausted
end

"""
    gmg_velocity_coarse_solve!(mg, e, rhs, counters)

Chebyshev velocity relaxation of the coarsest `K_vv e = rhs_v` from `e = 0`
until the velocity residual drops by `mg.coarse.drop`. Returns the status.
"""
function gmg_velocity_coarse_solve!(mg::JustRelax.GMGMultigrid, e, rhs, counters::JustRelax.GMGCounters)
    level, op = last(mg.hierarchy.levels), last(mg.operators)
    solver = mg.coarse
    gmg_zero!(e)
    JustRelax.gmg_prepare!(op, e)
    initial = gmg_velocity_norm(op, level, rhs)
    iszero(initial) && return JustRelax.gmg_converged
    for _ in 1:solver.max_sweeps
        gmg_velocity_smooth!(op, e, rhs, solver.smoother)
        gmg_velocity_residual!(op, level.residual, rhs, e)
        counters.coarse_sweeps += 1
        gmg_count!(counters, solver.smoother.substeps + 1, gmg_weight(mg, length(mg.hierarchy.levels)))
        current = gmg_velocity_norm(op, level, level.residual)
        isfinite(current) || return JustRelax.gmg_nonfinite
        current <= solver.drop * initial && return JustRelax.gmg_converged
    end
    return JustRelax.gmg_budget_exhausted
end

"""
    gmg_velocity_vcycle!(mg, l, e, rhs, counters)

One V-cycle for the velocity block `K_vv e = rhs_v` on level `l`, improving
`e` in place (`e.p` must be zero). Coarse corrections start from zero.
"""
function gmg_velocity_vcycle!(mg::JustRelax.GMGMultigrid, l::Int, e, rhs, counters::JustRelax.GMGCounters)
    levels, ops = mg.hierarchy.levels, mg.operators
    if l == length(levels)
        counters.coarse_status = gmg_velocity_coarse_solve!(mg, e, rhs, counters)
        return nothing
    end
    level, op = levels[l], ops[l]
    l > 1 && gmg_zero!(e)
    for _ in 1:mg.pre
        gmg_velocity_smooth!(op, e, rhs, mg.smoother)
        counters.sweeps += 1
        gmg_count!(counters, mg.smoother.substeps, gmg_weight(mg, l))
    end
    gmg_velocity_residual!(op, level.residual, rhs, e)
    gmg_count!(counters, 1, gmg_weight(mg, l))
    coarse = levels[l + 1]
    gmg_restrict!(coarse.rhs, level.residual, mg.transfers[l])
    gmg_velocity_vcycle!(mg, l + 1, coarse.correction, coarse.rhs, counters)
    counters.coarse_status == JustRelax.gmg_nonfinite && return nothing
    JustRelax.gmg_prepare!(ops[l + 1], coarse.correction)
    # Energy-minimizing step along the prolongated correction c (kept in op.direction):
    # a rediscretized coarse operator can overshoot under viscosity contrast.
    if mg.line_search
        c = op.direction
        gmg_prolong!(c, coarse.correction, mg.transfers[l]; β = 0)
        JustRelax.gmg_prepare!(op, c)
        gmg_apply_operator!(op, c)
        gmg_count!(counters, 1, gmg_weight(mg, l))
        α = gmg_energy_step(op, level, c, level.residual)
        foreach((v, dv) -> v .+= α .* dv, e.v, c.v)
    else
        gmg_prolong!(e, coarse.correction, mg.transfers[l])
    end
    JustRelax.gmg_prepare!(op, e)
    for _ in 1:mg.post
        gmg_velocity_smooth!(op, e, rhs, mg.smoother)
        counters.sweeps += 1
        gmg_count!(counters, mg.smoother.substeps, gmg_weight(mg, l))
    end
    return nothing
end

"""
    gmg_velocity_solve!(mg, e, rhs; rtol = 1e-8, max_cycles = 100, stagnation = 0.99)

V-cycle iteration for the finest velocity block `K_vv e = rhs_v` (`e.p` must
be zero). Stops as [`gmg_solve!`](@ref) does, on the velocity residual.
Returns `(status, counters, history)`.
"""
function gmg_velocity_solve!(mg::JustRelax.GMGMultigrid, e, rhs; rtol = 1.0e-8, max_cycles::Integer = 100, stagnation = 0.99)
    level, op = first(mg.hierarchy.levels), first(mg.operators)
    counters = JustRelax.GMGCounters()
    history = eltype(e.p)[]
    gmg_velocity_residual!(op, level.residual, rhs, e)
    previous = gmg_velocity_norm(op, level, level.residual)
    push!(history, previous)
    iszero(previous) && return JustRelax.gmg_converged, counters, history
    initial = previous
    for _ in 1:max_cycles
        gmg_velocity_vcycle!(mg, 1, e, rhs, counters)
        counters.cycles += 1
        counters.coarse_status == JustRelax.gmg_nonfinite && return JustRelax.gmg_nonfinite, counters, history
        gmg_velocity_residual!(op, level.residual, rhs, e)
        gmg_count!(counters, 1, 1)
        current = gmg_velocity_norm(op, level, level.residual)
        push!(history, current)
        isfinite(current) || return JustRelax.gmg_nonfinite, counters, history
        current <= rtol * initial && return JustRelax.gmg_converged, counters, history
        current > stagnation * previous && return JustRelax.gmg_stagnated, counters, history
        previous = current
    end
    return JustRelax.gmg_budget_exhausted, counters, history
end

"""
    gmg_ph_solve!(mg, e, rhs; rtol = 1e-8, max_iterations = 100, velocity_cycles = 1, velocity_rtol = 1e-1, pressure_step = 1, patience = 10)

Outer PH (inexact augmented Uzawa) iteration for the coupled `K e = rhs` on the
finest level, with velocity-block multigrid:

1. `δv ≈ K_vv⁻¹ (rhs_v - K_v e)` by [`gmg_velocity_fcg!`](@ref) (at most
   `velocity_cycles` V-cycles, or a velocity residual drop of `velocity_rtol`);
   `v += δv`.
2. `p += pressure_step · pressure_scale · (rhs_p - K_p e)` from fresh continuity.

`pressure_scale` is the operator coefficient (an inverse Schur estimate, e.g.
`η + γ`). `e` and `rhs` must not alias the finest level's workspaces. Stops on
a block-normalized residual drop of `rtol`, the iteration budget, `patience`
iterations without a new residual minimum (the PH iterate is not monotone),
or a nonfinite residual. Returns `(status, counters, history)`.
"""
function gmg_ph_solve!(
        mg::JustRelax.GMGMultigrid, e, rhs; rtol = 1.0e-8, max_iterations::Integer = 100,
        velocity_cycles::Integer = 1, velocity_rtol = 1.0e-1, pressure_step = 1, patience::Integer = 10
    )
    level, op = first(mg.hierarchy.levels), first(mg.operators)
    r, δ = level.rhs, level.correction
    counters = JustRelax.GMGCounters()
    T = eltype(e.p)
    β = T(pressure_step)
    history = T[]
    gmg_residual!(op, r, rhs, e)
    gmg_count!(counters, 1, 1)
    norms = gmg_block_norms(op, level, r)
    scales = gmg_scales(norms...)
    initial = gmg_combined(norms..., scales)
    push!(history, initial)
    iszero(initial) && return JustRelax.gmg_converged, counters, history
    best, since_best = initial, 0
    for _ in 1:max_iterations
        velocity = gmg_velocity_fcg!(mg, δ, r, counters; rtol = velocity_rtol, max_cycles = velocity_cycles)
        velocity == JustRelax.gmg_nonfinite && return JustRelax.gmg_nonfinite, counters, history
        foreach((v, dv) -> v .+= dv, e.v, δ.v)
        JustRelax.gmg_prepare!(op, e)
        gmg_apply_operator!(op, e)
        e.p .+= ifelse.(iszero.(op.pressure_mask), zero(T), β .* op.coefficients.pressure_scale .* (rhs.p .- op.out.p))
        gmg_residual!(op, r, rhs, e)
        counters.iterations += 1
        gmg_count!(counters, 2, 1)
        current = gmg_combined(gmg_block_norms(op, level, r)..., scales)
        push!(history, current)
        isfinite(current) || return JustRelax.gmg_nonfinite, counters, history
        current <= rtol * initial && return JustRelax.gmg_converged, counters, history
        best, since_best = current < best ? (current, 0) : (best, since_best + 1)
        since_best >= patience && return JustRelax.gmg_stagnated, counters, history
    end
    return JustRelax.gmg_budget_exhausted, counters, history
end
