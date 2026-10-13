"""Outcome of a bounded GMG solve; budget exhaustion is not success."""
@enum GMGStatus::Int8 gmg_converged gmg_budget_exhausted gmg_stagnated gmg_nonfinite

"""
    GMGCoarseSolver(; max_sweeps = 200, drop = 0.1, smoother = GMGSmoother(; spectral_range = 100))

Coarsest-level relaxation: DR/PH sweeps until the block-normalized coupled
residual drops by `drop`, or `max_sweeps` is reached.
"""
struct GMGCoarseSolver{T}
    max_sweeps::Int
    drop::T
    smoother::GMGSmoother{T}
end

function GMGCoarseSolver(; max_sweeps::Integer = 200, drop::Real = 0.1, smoother = GMGSmoother(; spectral_range = 100))
    max_sweeps >= 1 || throw(ArgumentError("coarse solver needs at least one sweep"))
    0 < drop < 1 || throw(ArgumentError("coarse residual drop must lie in (0, 1)"))
    T = typeof(smoother.spectral_range)
    return GMGCoarseSolver{T}(Int(max_sweeps), T(drop), smoother)
end

"""
Work counters of one solve: outer PH iterations, V-cycles, smoothing sweeps,
coarsest-level sweeps, operator applications (smoothing and residuals,
probing excluded), and their cost in finest-level applications (`work`).
"""
mutable struct GMGCounters
    iterations::Int
    cycles::Int
    sweeps::Int
    coarse_sweeps::Int
    applications::Int
    work::Float64
    coarse_status::GMGStatus
end

GMGCounters() = GMGCounters(0, 0, 0, 0, 0, 0.0, gmg_converged)

"""
    GMGMultigrid(hierarchy; velocity, no_slip, pre, post, smoother, coarse, line_search = true)

Linear multigrid workspace: transfers and matrix-free operators for every level
of `hierarchy`, the fine/intermediate smoother, `pre`/`post` sweep counts, the
coarsest-level relaxation, and finest-level Krylov scratch. `line_search`
scales each prolongated velocity-block correction by its energy-minimizing step. Coefficients are set on the finest operator and
restricted with `gmg_restrict_coefficients!`; bounds need `gmg_refresh_bounds!`.
"""
struct GMGMultigrid{H, X, O, T, F}
    hierarchy::H
    transfers::Vector{X}
    operators::Vector{O}
    smoother::GMGSmoother{T}
    coarse::GMGCoarseSolver{T}
    pre::Int
    post::Int
    line_search::Bool
    krylov::NTuple{3, F}
end

function GMGMultigrid(
        hierarchy::GMGHierarchy; velocity = GMGConservativeVelocity(), no_slip = nothing,
        pre::Integer = 1, post::Integer = 1, smoother = GMGSmoother(), coarse = GMGCoarseSolver(),
        line_search::Bool = true
    )
    levels = hierarchy.levels
    bcs = isnothing(no_slip) ? ntuple(_ -> (false, false), Val(length(first(levels).partition.global_ni))) : no_slip
    transfers = gmg_transfers(hierarchy; velocity, no_slip = bcs)
    operators = [GMGLevelOperator(level; no_slip = bcs) for level in levels]
    pre >= 0 && post >= 0 && pre + post >= 1 || throw(ArgumentError("a V-cycle needs at least one smoothing sweep"))
    fine = first(levels).correction.p
    krylov = ntuple(_ -> gmg_fields(fine, size(fine)), Val(3))
    return GMGMultigrid(hierarchy, transfers, operators, smoother, coarse, Int(pre), Int(post), line_search, krylov)
end
