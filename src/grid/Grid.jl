include("Utils.jl")

# MPI struct

struct IGG{T, M}
    me::T
    dims::Vector{T}
    nprocs::T
    coords::Vector{T}
    comm_cart::M
end

# Staggered grid

"""
    struct Geometry{nDim,T}

A struct representing the geometry of a topological object in nDim dimensions.

# Arguments
- `nDim`: The number of dimensions of the topological object.
- `T`: The type of the elements in the topological object.
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
    grid_v::NTuple{nDim, NTuple{nDim, V}} # velocity grid
end

# Default uniform staggered grid constructor
function Geometry(
        ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}; origin = ntuple(_ -> 0.0, Val(nDim))
    ) where {nDim, T}
    isMPI = ImplicitGlobalGrid.grid_is_initialized()

    Li, maxLi, di, xci, xvi, grid_v = if isMPI
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

    return Geometry{nDim, typeof(xci[1]), typeof(di), Float64}(ni, Li, origin, maxLi, di, _di, xci, xvi, grid_v)
end

# Grid constructor given 1D vertex coordinates arrays
function Geometry(xvi::Vararg{T, nDim}) where {nDim, T <: AbstractVector}

    ni = length.(xvi) .- 1
    xci = ntuple(i -> [(xvi[i][j] + xvi[i][j + 1]) / 2 for j in 1:ni[i]], Val(nDim))
    xci = ntuple(Val(nDim)) do i
        @views @. (xvi[i][1:(end - 1)] + xvi[i][2:end]) / 2
    end
    lims = extrema.(xvi)
    li = ntuple(i -> lims[i][2] - lims[i][1], Val(nDim))
    max_li = reduce(max, li)
    origin = ntuple(i -> lims[i][1], Val(nDim))
    di_vertex = diff.(xvi)
    di_center = diff.(xci)
    grid_v = velocity_grids(xci, xvi, di_center)
    di_vel = ntuple(i -> diff.(grid_v[i]), Val(nDim))
    di = (; center = di_center, vertex = di_vertex, velocity = di_vel)
    _di = (;
        center = map(x -> inv.(x), di.center),
        vertex = map(x -> inv.(x), di.vertex),
        velocity = map(x -> map(y -> inv.(y), x), di.velocity),
    )

    return Geometry{nDim, typeof(xci[1]), typeof(di), Float64}(ni, li, origin, max_li, di, _di, xci, xvi, grid_v)
end

Geometry(xvi::NTuple{nDim, T}) where {nDim, T <: AbstractVector} = Geometry(xvi...)

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
    grid_v = velocity_grids(xci, xvi, di)
    return Li, max(Li...), di, xci, xvi, grid_v
end

function geometry_nonMPI(
        ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}, origin
    ) where {nDim, T}
    Li = Float64.(li)
    di = Li ./ ni

    xci, xvi = lazy_grid(di, ni, Li; origin = origin)
    grid_v = velocity_grids(xci, xvi, di)
    return Li, max(Li...), di, xci, xvi, grid_v
end

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
    velocity_grids(xci, xvi, di::NTuple{N,T}) where {N,T}

Compute the velocity grids for N dimensionional problems.

# Arguments
- `xci`: The x-coordinate of the cell centers.
- `xvi`: The x-coordinate of the cell vertices.
- `di`: A tuple containing the cell dimensions.
"""
function velocity_grids(xci, xvi, di::NTuple{2, T}) where {T}
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

function velocity_grids(xci, xvi, di::NTuple{3, T}) where {T}
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
