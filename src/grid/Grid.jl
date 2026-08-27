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

abstract type AbstractGrid{nDim, V, D, T} end
include("Cartesian.jl")
include("Annulus.jl")

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
