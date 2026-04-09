include("Utils.jl")

# MPI struct
"""
    IGG(me, dims, nprocs, coords, comm_cart)

Container for the Cartesian MPI topology returned by `ImplicitGlobalGrid.init_global_grid`.

This is typically created as:

```julia
igg = IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
```

and then passed around so code can access the current rank, Cartesian coordinates,
and communicator associated with the distributed grid decomposition.
"""
struct IGG{T, M}
    me::T
    dims::Vector{T}
    nprocs::T
    coords::Vector{T}
    comm_cart::M
end

# Staggered grid

"""
    struct Geometry{nDim,V,D,T}

A staggered Cartesian grid in `nDim` dimensions.

`Geometry` stores the domain size, origin, cell spacing, cell-centered coordinates,
vertex coordinates, and the staggered velocity grids used throughout JustRelax.
"""
struct Geometry{nDim, V, D, T}
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
    isMPI = ImplicitGlobalGrid.grid_is_initialized()

    Li, maxLi, di, xci, xvi, xi_vel = if isMPI
        geometry_MPI(ni, li, origin)
    else
        geometry_nonMPI(ni, li, origin)
    end

    di = (; center = di, vertex = di, velocity = ntuple(i -> di, Val(nDim)))
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

Geometry(xvi::NTuple{nDim, T}) where {nDim, T <: AbstractVector} = Geometry(xvi...)

"""
    legacy_uniform_grid(ni, di)

Construct a uniform [`Geometry`](@ref) from grid sizes `ni` and cell spacings `di`.

This helper preserves the older API used by some solver code. In MPI mode the physical
domain lengths are reconstructed from the global grid dimensions, so the resulting
geometry matches the full distributed domain rather than only the local chunk.
"""
@inline function legacy_uniform_grid(
        ni::NTuple{nDim, <:Integer}, di::NTuple{nDim, <:Real}
    ) where {nDim}
    ni_global = if ImplicitGlobalGrid.grid_is_initialized()
        ntuple(i -> (nx_g, ny_g, nz_g)[i](), Val(nDim))
    else
        ni
    end
    li = ntuple(i -> Float64(di[i]) * ni_global[i], Val(nDim))
    return Geometry(ni, li)
end

@inline legacy_uniform_grid(ni::NTuple{N, <:Integer}, di::NamedTuple) where {N} =
    legacy_uniform_grid(ni, di.center)

function geometry_MPI(ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}, origin) where {nDim, T}
    f_g = (nx_g, ny_g, nz_g)
    ni_g = ntuple(i -> f_g[i](), Val(nDim))
    Li = Float64.(li)
    di = Li ./ ni_g
    xci, xvi = lazy_grid_MPI(di, ni; origin = origin)
    xi_vel = velocity_grids(xci, xvi, di)
    return Li, max(Li...), di, xci, xvi, xi_vel
end

function geometry_nonMPI(
        ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}, origin
    ) where {nDim, T}
    Li = Float64.(li)
    di = Li ./ ni

    xci, xvi = lazy_grid(di, ni, Li; origin = origin)
    xi_vel = velocity_grids(xci, xvi, di)
    return Li, max(Li...), di, xci, xvi, xi_vel
end

"""
    lazy_grid_MPI(di, ni; origin = ntuple(_ -> zero(T1), Val(N)))

Create local cell-centered and vertex coordinates for a uniform grid distributed with
`ImplicitGlobalGrid`.

The returned coordinates are shifted by `origin` and correspond to the local MPI rank.
"""
function lazy_grid_MPI(
        di::NTuple{N, T1}, ni; origin = ntuple(_ -> zero(T1), Val(N))
    ) where {N, T1}
    f_g = (x_g, y_g, z_g)

    # nodes at the center of the grid cells
    xci = ntuple(Val(N)) do i
        Base.@_inline_meta
        rank_origin = f_g[i](1, di[i], ni[i])

        local_origin = rank_origin + origin[i]
        rank_end = f_g[i](ni[i], di[i], ni[i])
        local_end = rank_end + origin[i]

        @inbounds LinRange(local_origin[i] + di[i] / 2, local_end[i] + di[i] / 2, ni[i])
    end

    # nodes at the vertices of the grid cells
    xvi = ntuple(Val(N)) do i
        # println("potato")
        Base.@_inline_meta
        rank_origin = f_g[i](1, di[i], ni[i])
        local_origin = rank_origin + origin[i]

        rank_end = f_g[i](ni[i] + 1, di[i], ni[i])
        local_end = rank_end + origin[i]

        @inbounds LinRange(local_origin[i], local_end[i], ni[i] + 1)
    end

    return xci, xvi
end

"""
    lazy_grid(di, ni, Li; origin = ntuple(_ -> zero(T1), Val(N)))

Create cell-centered and vertex coordinates for a serial uniform grid.

`di` gives the spacing in each direction, `ni` the number of cells, and `Li` the
physical lengths of the domain.
"""
function lazy_grid(
        di::NTuple{N, T1}, ni, Li; origin = ntuple(_ -> zero(T1), Val(N))
    ) where {N, T1}

    # nodes at the center of the grid cells
    xci = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds LinRange(origin[i] + di[i] / 2, origin[i] + Li[i] - di[i] / 2, ni[i])
    end

    # nodes at the vertices of the grid cells
    xvi = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds LinRange(origin[i], origin[i] + Li[i], ni[i] + 1)
    end

    return xci, xvi
end

# Velocity helper grids for the particle advection

"""
    velocity_grids(xci, xvi, di)

Build staggered velocity coordinates from cell-centered and vertex grids.

For each velocity component, the coordinate along that component lives on vertices,
while the transverse directions are extended with one ghost point on either side.
Both uniform spacings and nonuniform spacing vectors are supported in 2D and 3D.

# Arguments
- `xci`: Cell-centered coordinates in each direction.
- `xvi`: Vertex coordinates in each direction.
- `di`: Cell spacing as either scalars for a uniform grid or vectors for a nonuniform grid.
"""
function velocity_grids(xci, xvi, di::NTuple{2, Number})
    dx, dy = @dxi(di, 1, 1)
    yVx = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2]) + 2)
    xVy = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1]) + 2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

function velocity_grids(xci, xvi, di::NTuple{2, T}) where {T <: AbstractVector}
    dxW, dyW = @dxi(di, 1, 1)
    dxE, dyE = @dxi(di, length.(di)...)

    xghost = vcat(xci[1][1] - dxW, xci[1], xci[1][end] + dxE)
    yghost = vcat(xci[2][1] - dyW, xci[2], xci[2][end] + dyE)

    grid_vx = xvi[1], yghost
    grid_vy = xghost, xvi[2]

    return grid_vx, grid_vy
end

function velocity_grids(xci, xvi, di::NTuple{3, Number})
    xghost = ntuple(Val(3)) do i
        dii = if i == 1
            @dx(di, 1)
        elseif i == 2
            @dy(di, 1)
        else
            @dz(di, 1)
        end
        return LinRange(xci[i][1] - dii, xci[i][end] + dii, length(xci[i]) + 2)
    end
    grid_vx = xvi[1], xghost[2], xghost[3]
    grid_vy = xghost[1], xvi[2], xghost[3]
    grid_vz = xghost[1], xghost[2], xvi[3]

    return grid_vx, grid_vy, grid_vz
end

function velocity_grids(xci, xvi, di::NTuple{3, T}) where {T <: AbstractVector}
    dxW, dyW, dzW = @dxi(di, 1, 1, 1)
    dxE, dyE, dzE = @dxi(di, length.(di)...)

    xghost = vcat(xci[1][1] - dxW, xci[1], xci[1][end] + dxE)
    yghost = vcat(xci[2][1] - dyW, xci[2], xci[2][end] + dyE)
    zghost = vcat(xci[3][1] - dzW, xci[3], xci[3][end] + dzE)

    grid_vx = xvi[1], yghost, zghost
    grid_vy = xghost, xvi[2], zghost
    grid_vz = xghost, yghost, xvi[3]

    return grid_vx, grid_vy, grid_vz
end
