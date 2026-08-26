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
    Geometry(TA, xvi::Vararg{<:AbstractVector, nDim})
    Geometry(xvi::NTuple{nDim, <:AbstractVector})

Build a staggered grid from explicit vertex coordinates along each dimension.

This constructor is useful for refined or otherwise nonuniform meshes. Cell-centered
coordinates, local spacings, and staggered velocity grids are derived from the supplied
vertex coordinates. `TA` can be used to move the generated arrays to a target array type.

# Arguments
- `TA`: Array constructor used to materialize the coordinate arrays, for example `Array`
  or a backend-specific array type.
- `xvi`: One vertex-coordinate vector per dimension.
"""
function Geometry(TA::Type{A}, xvi::Vararg{T, nDim}) where {nDim, A <: AbstractArray, T <: AbstractVector}
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
