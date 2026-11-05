"""
    struct Geometry{nDim,V,D,T}

A staggered Cartesian grid in `nDim` dimensions.

`Geometry` stores the domain size, origin, cell spacing, cell-centered coordinates,
vertex coordinates, and the staggered velocity grids used throughout JustRelax.
"""
struct Geometry{nDim, V, D, T} <: AbstractGrid{nDim, V, D, T}
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

# Default uniform staggered grid constructor
"""
    Geometry(ni, li; origin = ntuple(_ -> 0.0, Val(nDim)))

Build a uniform staggered grid with `ni` cells and physical domain lengths `li`.

When `ImplicitGlobalGrid` has been initialized, the grid spacing is computed from
the global grid dimensions and the returned coordinates correspond to the local MPI
subdomain. Otherwise a serial grid covering the full domain is created.

# Arguments
- `ni`: Number of local grid cells in each direction.
- `li`: Physical domain length in each direction.

# Keywords
- `origin`: Lower-left or lower-front corner of the domain.

# Returns
- A [`Geometry`](@ref) with cell-centered coordinates `xci`, vertex coordinates `xvi`,
  and staggered velocity coordinates `xi_vel`.
"""
function Geometry(
        ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}; origin = ntuple(_ -> 0.0, Val(nDim))
    ) where {nDim, T}
    Li, maxLi, di, xci, xvi, xi_vel = if ImplicitGlobalGrid.grid_is_initialized()
        geometry_MPI(ni, li, origin)
    else
        geometry_nonMPI(ni, li, origin)
    end
    di = (; center = di, vertex = di, velocity = ntuple(_ -> di, Val(nDim)))
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )
    return Geometry{nDim, typeof(xci[1]), typeof(di), Float64}(ni, Li, origin, maxLi, di, _di, xci, xvi, xi_vel)
end

# Grid constructor given 1D vertex coordinates arrays
"""
    Geometry(TA, xvi::Vararg{<:AbstractVector, nDim}; distributed = ImplicitGlobalGrid.grid_is_initialized())
    Geometry(xvi::NTuple{nDim, <:AbstractVector})

Build a staggered grid from explicit vertex coordinates along each dimension.

This constructor is useful for refined or otherwise nonuniform meshes. Cell-centered
coordinates, local spacings, and staggered velocity grids are derived from the supplied
vertex coordinates. `TA` can be used to move the generated arrays to a target array type.

When `distributed` is `true` (the default once `ImplicitGlobalGrid` has been initialized),
`xvi` are the vertex coordinates of the global grid and the returned grid is the subdomain
of the local MPI rank, including its overlap cells. As for the uniform constructor, `li`,
`max_li` and `origin` describe the global domain. The ghost nodes of the staggered velocity
grids, and the spacings derived from them, lie at the coordinates of the neighboring cells,
so a refined grid is continuous across ranks. Pass `distributed = false` to build the
global grid on every rank.

# Arguments
- `TA`: Array constructor used to materialize the coordinate arrays, for example `Array`
  or a backend-specific array type.
- `xvi`: One vertex-coordinate vector per dimension; of the global grid when distributed.

# Keyword Arguments
- `distributed`: Return the local subdomain of the MPI rank. Default:
  `ImplicitGlobalGrid.grid_is_initialized()`.
"""
function Geometry(
        TA::Type{A}, xvi::Vararg{T, nDim}; distributed::Bool = ImplicitGlobalGrid.grid_is_initialized()
    ) where {nDim, A <: AbstractArray, T <: AbstractVector}
    grid = _geometry_from_vertices(TA, xvi...)
    return distributed ? local_subdomain(grid) : grid
end

function _geometry_from_vertices(TA::Type{A}, xvi::Vararg{T, nDim}) where {nDim, A <: AbstractArray, T <: AbstractVector}
    ni = length.(xvi) .- 1
    xci = ntuple(Val(nDim)) do i
        @views @. (xvi[i][1:(end - 1)] + xvi[i][2:end]) / 2
    end
    lims = extrema.(xvi)
    li = ntuple(i -> lims[i][2] - lims[i][1], Val(nDim))
    max_li = reduce(max, li)
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
    return Geometry{nDim, eltype(xi_vel[1]), typeof(di), Float64}(ni, li, origin, max_li, di, _di, TA.(xci), TA.(xvi), xi_vel)
end

Geometry(xvi::NTuple{nDim, T}) where {nDim, T <: AbstractVector} = Geometry(Array, xvi...)

# Subdomain of the local MPI rank, sliced out of the global grid `grid`. ImplicitGlobalGrid
# places local index `i` of the rank at Cartesian coordinate `c` on global index
# `c * (n - overlap) + i`, where `n` is the local number of cells. Every coordinate and
# spacing array along a dimension holds the number of cells plus a fixed number of entries,
# so the same offset slices all of them.
function local_subdomain(grid::Geometry{nDim}) where {nDim}
    ImplicitGlobalGrid.grid_is_initialized() ||
        throw(ArgumentError("a distributed `Geometry` requires an ImplicitGlobalGrid; call `init_global_grid` first"))
    gg = ImplicitGlobalGrid.global_grid()
    ni_g = ntuple(d -> gg.nxyz_g[d], Val(nDim))
    grid.ni == ni_g || throw(
        DimensionMismatch(
            "the vertex coordinates describe $(grid.ni) cells, but the global grid has $ni_g cells; pass the coordinates of the global grid"
        )
    )
    any(d -> !iszero(gg.periods[d]), 1:nDim) &&
        throw(ArgumentError("a nonuniform `Geometry` does not support periodic MPI directions"))
    ni = ntuple(d -> gg.nxyz[d], Val(nDim))
    offset = ntuple(d -> gg.coords[d] * (gg.nxyz[d] - gg.overlaps[d]), Val(nDim))
    slice(v, d) = v[offset[d] .+ (1:(length(v) - ni_g[d] + ni[d]))]
    slice_dims(t) = ntuple(d -> slice(t[d], d), Val(nDim))
    di = (;
        center = slice_dims(grid.di.center),
        vertex = slice_dims(grid.di.vertex),
        velocity = map(slice_dims, grid.di.velocity),
    )
    _di = (;
        center = slice_dims(grid._di.center),
        vertex = slice_dims(grid._di.vertex),
        velocity = map(slice_dims, grid._di.velocity),
    )
    return typeof(grid)(
        ni, grid.li, grid.origin, grid.max_li, di, _di,
        slice_dims(grid.xci), slice_dims(grid.xvi), map(slice_dims, grid.xi_vel),
    )
end
