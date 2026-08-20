"""
    GeometryAnnulus{nDim,V,D,T} <: AbstractGrid{nDim,V,D,T}

A staggered annular grid whose first two coordinate directions are `(θ, r)`: angular
position followed by radius. Accordingly, `xci[1]` and `xvi[1]` contain angular
coordinates in radians, while `xci[2]` and `xvi[2]` contain radial coordinates.

The grid stores its cell counts, coordinate extents, origin, spacings, cell-centered
coordinates, vertex coordinates, and staggered velocity grids.
"""
struct GeometryAnnulus{nDim, V, D, T} <: AbstractGrid{nDim, V, D, T}
    ni::NTuple{nDim, Int64}               # number of grid cells
    li::NTuple{nDim, T}                   # length of the grid
    origin::NTuple{nDim, T}               # origin of the grid
    max_li::T                             # maximum length of the grid
    di::D                                 # grid spacing
    _di::D                                # inverse grid spacing
    xci::NTuple{nDim, V}                  # cell-centered grid
    xvi::NTuple{nDim, V}                  # vertex-centered grid
    xi_vel::NTuple{nDim, NTuple{nDim, V}} # velocity grid
end

"""
    GeometryAnnulus(ni, li; origin = ntuple(_ -> 0.0, Val(nDim)))

Build a uniform staggered annular grid in `(θ, r)` coordinate order.

For a two-dimensional annulus, `ni = (nθ, nr)`, `li = (lθ, lr)`, and
`origin = (θ₀, r₀)`. Thus, the first entries describe the angular direction and the
second entries describe the radial direction. Both `lθ` and `θ₀` must be given in
radians. When `ImplicitGlobalGrid` is initialized, the coordinates cover the local MPI
subdomain; otherwise they cover the full domain.

# Arguments
- `ni`: Number of cells in each coordinate direction, ordered as `(nθ, nr)`.
- `li`: Coordinate extents, ordered as angular extent then radial extent.

# Keywords
- `origin`: Lower coordinate bounds, ordered as `(θ₀, r₀)`.

# Returns
- A [`GeometryAnnulus`](@ref) with cell-centered, vertex, and staggered velocity
  coordinates in `(θ, r)` order.
"""
function GeometryAnnulus(
        ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}; origin = ntuple(_ -> 0.0, Val(nDim))
    ) where {nDim, T}
    Li, _, di, xci, xvi, xi_vel = if ImplicitGlobalGrid.grid_is_initialized()
        geometry_MPI(ni, li, origin)
    else
        geometry_nonMPI(ni, li, origin)
    end

    maxLi = max(prod(li), li[2])
    di = (; center = di, vertex = di, velocity = ntuple(_ -> di, Val(nDim)))
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )
    return GeometryAnnulus{nDim, typeof(xci[1]), typeof(di), Float64}(ni, Li, origin, maxLi, di, _di, xci, xvi, xi_vel)
end

"""
    GeometryAnnulus(TA, θv, rv)
    GeometryAnnulus((θv, rv))

Build a staggered annular grid from explicit vertex-coordinate vectors in `(θ, r)`
order. `θv` contains angular vertices and `rv` contains radial vertices. Cell-centered
coordinates and local spacings are derived from these vectors. All values in `θv` must
be given in radians.

`TA` is the array constructor used to materialize the generated coordinate arrays,
such as `Array` or a backend-specific array type. The tuple form uses `Array`.
"""
function GeometryAnnulus(TA::Type{A}, xvi::Vararg{T, nDim}) where {nDim, A <: AbstractArray, T <: AbstractVector}
    ni = length.(xvi) .- 1
    xci = ntuple(Val(nDim)) do i
        @views @. (xvi[i][1:(end - 1)] + xvi[i][2:end]) / 2
    end
    lims = extrema.(xvi)
    li = ntuple(i -> lims[i][2] - lims[i][1], Val(nDim))
    max_li = max(prod(li), li[2])
    origin = ntuple(i -> lims[i][1], Val(nDim))
    di_vertex = diff.(xvi)
    di_center = diff.(xci)
    xi_vel_cpu = velocity_grids(xci, xvi, di_center)
    xi_vel = ntuple(i -> TA.(xi_vel_cpu[i]), Val(nDim))
    di_vel = ntuple(i -> diff.(xi_vel[i]), Val(nDim))
    di = (; center = TA.(di_center), vertex = TA.(di_vertex), velocity = di_vel)
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )
    return GeometryAnnulus{nDim, eltype(xi_vel[1]), typeof(di), Float64}(ni, li, origin, max_li, di, _di, TA.(xci), TA.(xvi), xi_vel)
end

GeometryAnnulus(xvi::NTuple{nDim, T}) where {nDim, T <: AbstractVector} = GeometryAnnulus(Array, xvi...)
