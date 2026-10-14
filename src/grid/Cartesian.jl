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

"""
    Geometry(TA, xvi_global::NTuple, igg::IGG)
    Geometry(xvi_global::NTuple, igg::IGG)

Partition explicit global vertex coordinates onto the initialized IGG topology.
Supports nonperiodic grids with the default two-cell overlap. The coordinate
lengths must match the global cell counts plus one. Global domain lengths are
retained for solver scaling; `origin`, `xci`, `xvi`, and `xi_vel` describe the local
rank. Transverse velocity ghosts use the actual neighboring global coordinates.
`TA` selects the coordinate array type and defaults to `Array`.
"""
function Geometry(TA::Type{<:AbstractArray}, xvi_global::NTuple{N, AbstractVector}, igg::IGG) where {N}
    ImplicitGlobalGrid.grid_is_initialized() || throw(ArgumentError("initialize IGG before constructing a distributed Geometry"))
    gg = ImplicitGlobalGrid.global_grid()
    all(==(2), gg.overlaps[1:N]) && all(iszero, gg.periods[1:N]) ||
        throw(ArgumentError("explicit-coordinate distributed Geometry requires overlap=2 and nonperiodic boundaries"))
    length.(xvi_global) == Tuple(gg.nxyz_g[1:N]) .+ 1 ||
        throw(DimensionMismatch("global vertex coordinates must match the IGG global cell counts plus one"))
    ni = Tuple(gg.nxyz[1:N])
    ranges = ntuple(d -> (igg.coords[d] * (ni[d] - 2) + 1):(igg.coords[d] * (ni[d] - 2) + ni[d] + 1), N)
    vertices = Array.(xvi_global)
    local_grid = Geometry(TA, ntuple(d -> vertices[d][ranges[d]], N)...)
    global_grid = Geometry(Array, vertices...)
    xi_vel = ntuple(N) do component
        ntuple(N) do d
            r = ranges[d]
            indices = component == d ? r : first(r):(last(r) + 1)
            TA(global_grid.xi_vel[component][d][indices])
        end
    end
    di = merge(local_grid.di, (; velocity = map(v -> diff.(v), xi_vel)))
    _di = merge(local_grid._di, (; velocity = map(v -> map(x -> inv.(x), v), di.velocity)))
    return typeof(local_grid)(
        ni, global_grid.li, local_grid.origin, global_grid.max_li,
        di, _di, local_grid.xci, local_grid.xvi, xi_vel,
    )
end

Geometry(xvi_global::NTuple{N, AbstractVector}, igg::IGG) where {N} = Geometry(Array, xvi_global, igg)
