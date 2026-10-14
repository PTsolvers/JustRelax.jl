# The optional extension supplies only the axis solver.
function _solve_refined_axis end

function _root_value(f, igg::IGG)
    result = if igg.me == 0
        try
            (f(), nothing)
        catch err
            (nothing, err)
        end
    else
        nothing
    end
    value, failure = MPI.bcast(result, 0, igg.comm_cart)
    isnothing(failure) || throw(failure)
    return value
end

"""
    Geometry(TA, ni, li, monitors; origin=ntuple(_ -> 0.0, length(ni)), igg=nothing)
    Geometry(ni, li, monitors; kwargs...)

Generate a rectilinear grid. Nonuniform monitors require `using PoissonGrids`.
`monitors` is a tuple of scalar monitor functions, one per axis; `nothing`
selects uniform spacing on that axis. Monitor coordinates use the same units as
`li` and `origin`. PoissonGrids solver validation and convergence errors propagate.

Without `igg`, `ni` counts global cells and the full domain is returned, even
when IGG is initialized. With `igg`, construction is collective and every rank
must call it; `ni` must match its local cell counts.
Cartesian rank zero generates the global coordinates and broadcasts them before
partitioning. Grid-generation errors propagate to every rank.
Distributed grids require nonperiodic boundaries and the default two-cell overlap.
`TA` selects the coordinate array type and defaults to `Array`.
"""
function Geometry(
        TA::Type{<:AbstractArray}, ni::NTuple{N, Integer}, li::NTuple{N, Real}, monitors::NTuple{N, Any};
        origin = ntuple(_ -> 0.0, N), igg::Union{Nothing, IGG} = nothing,
    ) where {N}
    ng = if isnothing(igg)
        ni
    else
        ImplicitGlobalGrid.grid_is_initialized() || throw(ArgumentError("initialize IGG before constructing a distributed Geometry"))
        gg = ImplicitGlobalGrid.global_grid()
        MPI.Allreduce(ni == Tuple(gg.nxyz[1:N]), &, igg.comm_cart) ||
            throw(DimensionMismatch("ni must match the IGG local cell counts on every rank"))
        Tuple(gg.nxyz_g[1:N])
    end
    vertices = refined_grid_vertices(ng, li, monitors; origin, igg)
    return isnothing(igg) ? Geometry(TA, vertices...) : Geometry(TA, vertices, igg)
end

"""
    refined_grid_vertices(ni, li, monitors; origin=ntuple(_ -> 0.0, length(ni)), igg=nothing)

Generate global CPU vertex-coordinate vectors. Nonuniform monitors require the
optional PoissonGrids extension (`using PoissonGrids`). `ni` always counts global
cells. Each axis takes a scalar monitor function, or `nothing` for uniform spacing, in the coordinate
units of `li` and `origin`.

With `igg`, this is collective: only Cartesian rank zero evaluates the monitors
and generates the coordinates, then broadcasts the vectors to every rank. Rank
zero's grid parameters are authoritative. Generation errors are rethrown on all
ranks. Without `igg`, generation is serial and errors propagate directly.

Use `Geometry(TA, vertices, igg)` to construct each rank's local staggered grid.
The global vectors can also be retained for distributed surface sampling.
"""
function refined_grid_vertices(
        ni::NTuple{N, Integer}, li::NTuple{N, Real}, monitors::NTuple{N, Any};
        origin = ntuple(_ -> 0.0, N), igg::Union{Nothing, IGG} = nothing,
    ) where {N}
    function generate()
        all(n -> n >= 2, ni) || throw(ArgumentError("each axis requires at least two cells"))
        all(x -> isfinite(x) && x > 0, li) || throw(ArgumentError("domain lengths must be finite and positive"))
        length(origin) == N && all(isfinite, origin) || throw(ArgumentError("origin must have one finite coordinate per axis"))
        return ntuple(N) do d
            lo, hi = Float64(origin[d]), Float64(origin[d] + li[d])
            isnothing(monitors[d]) ? collect(range(lo, hi; length = ni[d] + 1)) :
                _solve_refined_axis(lo, hi, monitors[d], ni[d])
        end
    end
    isnothing(igg) && return generate()
    return _root_value(generate, igg)
end

Geometry(ni::NTuple{N, Integer}, li::NTuple{N, Real}, monitors::NTuple{N, Any}; kwargs...) where {N} =
    Geometry(Array, ni, li, monitors; kwargs...)
