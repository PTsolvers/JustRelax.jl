module JustRelaxPoissonGridsExt

using JustRelax
using PoissonGrids

"""
    Geometry(TA, ni, li, monitors; origin=ntuple(_ -> 0.0, length(ni)), igg=nothing)
    Geometry(ni, li, monitors; kwargs...)

Generate a rectilinear grid with PoissonGrids. Requires `using PoissonGrids`.
`monitors` is a tuple of scalar monitor functions, one per axis; `nothing`
selects uniform spacing on that axis. Monitor coordinates use the same units as
`li` and `origin`. PoissonGrids solver validation and convergence errors propagate.

Without `igg`, `ni` counts global cells and the full domain is returned, even
when IGG is initialized. With `igg`, `ni` must match its local cell counts;
coordinates are generated at the global resolution and partitioned internally.
Distributed grids require nonperiodic boundaries and the default two-cell overlap.
`TA` selects the coordinate array type and defaults to `Array`.
"""
function JustRelax.Geometry(
        TA::Type{<:AbstractArray}, ni::NTuple{N, Integer}, li::NTuple{N, Real}, monitors::NTuple{N, Any};
        origin = ntuple(_ -> 0.0, N), igg::Union{Nothing, IGG} = nothing,
    ) where {N}
    all(n -> n >= 2, ni) || throw(ArgumentError("each axis requires at least two cells"))
    all(x -> isfinite(x) && x > 0, li) || throw(ArgumentError("domain lengths must be finite and positive"))
    length(origin) == N && all(isfinite, origin) || throw(ArgumentError("origin must have one finite coordinate per axis"))
    ng = if isnothing(igg)
        ni
    else
        JustRelax.ImplicitGlobalGrid.grid_is_initialized() || throw(ArgumentError("initialize IGG before constructing a distributed Geometry"))
        gg = JustRelax.ImplicitGlobalGrid.global_grid()
        ni == Tuple(gg.nxyz[1:N]) || throw(DimensionMismatch("ni must match the IGG local cell counts"))
        Tuple(gg.nxyz_g[1:N])
    end
    vertices = ntuple(N) do d
        lo, hi = Float64(origin[d]), Float64(origin[d] + li[d])
        isnothing(monitors[d]) ? collect(range(lo, hi; length = ng[d] + 1)) :
            solve_grid(lo, hi, monitors[d], ng[d])
    end
    return isnothing(igg) ? Geometry(TA, vertices...) : Geometry(TA, vertices, igg)
end

JustRelax.Geometry(ni::NTuple{N, Integer}, li::NTuple{N, Real}, monitors::NTuple{N, Any}; kwargs...) where {N} =
    Geometry(Array, ni, li, monitors; kwargs...)

end
