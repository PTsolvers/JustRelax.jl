abstract type AbstractGrid end

# Staggered grid

struct Geometry{nDim,T} <: AbstractGrid
    ni::NTuple{nDim,Int64}                      # number of grid cells
    li::NTuple{nDim,T}                          # length of the grid
    max_li::T                                   # maximum length of the grid
    di::NTuple{nDim,Vector{T}}                  # grid spacing
    _di::NTuple{nDim,Vector{T}}                 # grid spacing
    xci::NTuple{nDim,Vector{T}}                 # cell-centered grid
    xvi::NTuple{nDim,Vector{T}}                 # vertex-centered grid
    grid_v::NTuple{nDim,NTuple{nDim,Vector{T}}} # velocity grid

    function Geometry(xvi::NTuple{nDim,Vector{T}}) where {nDim,T}
        ni = @. length(xvi) - 1
        li = ntuple(Val(nDim)) do i
            l, r = extrema(xvi)
            abs(r - l)
        end
        max_li = max(li...)
        di = diff(xvi)
        _di = inv.(di)
        xci = ntuple(Val(nDim)) do i
            [(xvi[j + 1] - xvi[j]) / 2 for j in 1:ni[i]]
        end

        grid_v = velocity_grids(xci, xvi, di)
        return new{nDim,Float64}(ni, Li, origin, max(Li...), di, _di, xci, xvi, grid_v)
    end
end

"""
    struct Geometry{nDim,T}

A struct representing the geometry of a topological object in nDim dimensions.

# Arguments
- `nDim`: The number of dimensions of the topological object.
- `T`: The type of the elements in the topological object.
"""
struct LazyGeometry{nDim,T} <: AbstractGrid
    ni::NTuple{nDim,Int64}                              # number of grid cells
    li::NTuple{nDim,T}                                  # length of the grid
    origin::NTuple{nDim,T}                              # origin of the grid
    max_li::T                                           # maximum length of the grid
    di::NTuple{nDim,T}                                  # grid spacing
    _di::NTuple{nDim,T}                                  # grid spacing
    xci::NTuple{nDim,LinRange{T,Int64}}                 # cell-centered grid
    xvi::NTuple{nDim,LinRange{T,Int64}}                 # vertex-centered grid
    grid_v::NTuple{nDim,NTuple{nDim,LinRange{T,Int64}}} # velocity grid

    function LazyGeometry(
        ni::NTuple{nDim,Integer}, li::NTuple{nDim,T}; origin=ntuple(_ -> 0.0, Val(nDim))
    ) where {nDim,T}
        f_g = (nx_g, ny_g, nz_g)
        ni_g = ntuple(i -> f_g[i](), Val(nDim))
        Li = Float64.(li)
        di = Li ./ ni_g
        _di = inv.(di)

        xci, xvi = lazy_grid(di, ni; origin=origin)
        grid_v = velocity_grids(xci, xvi, di)
        return new{nDim,Float64}(ni, Li, origin, max(Li...), di, _di, xci, xvi, grid_v)
    end
end

function lazy_grid(di::NTuple{N,T1}, ni; origin=ntuple(_ -> zero(T1), Val(N))) where {N,T1}
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

# Velocity helper grids for the particle advection

"""
    velocity_grids(xci, xvi, di::NTuple{N,T}) where {N,T}

Compute the velocity grids for N dimensionional problems.

# Arguments
- `xci`: The x-coordinate of the cell centers.
- `xvi`: The x-coordinate of the cell vertices.
- `di`: A tuple containing the cell dimensions.
"""
function velocity_grids(xci, xvi, di::NTuple{2,T}) where {T}
    dx, dy = di
    yVx = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2]) + 2)
    xVy = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1]) + 2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

function velocity_grids(xci, xvi, di::NTuple{3,T}) where {T}
    xghost = ntuple(Val(3)) do i
        LinRange(xci[i][1] - di[i], xci[i][end] + di[i], length(xci[i]) + 2)
    end
    grid_vx = xvi[1], xghost[2], xghost[3]
    grid_vy = xghost[1], xvi[2], xghost[3]
    grid_vz = xghost[1], xghost[2], xvi[3]

    return grid_vx, grid_vy, grid_vz
end

function velocity_grids(
    xci::NTuple{2,AbstrctVector}, xvi::NTuple{2,AbstrctVector}, di::NTuple{2}
)
    xghost = ntuple(Val(2)) do i
        l = xci[i][1] - di[i][1]
        r = xci[i][end] + di[i][end]
        vcat(l, xci[i], r)
    end
    grid_vx = xvi[1], xghost[2]
    grid_vy = xghost[2], xvi[2]

    return grid_vx, grid_vy
end

function velocity_grids(
    xci::NTuple{3,AbstrctVector}, xvi::NTuple{3,AbstrctVector}, di::NTuple{3}
)
    xghost = ntuple(Val(3)) do i
        l = xci[i][1] - di[i][1]
        r = xci[i][end] + di[i][end]
        vcat(l, xci[i], r)
    end
    grid_vx = xvi[1], xghost[2], xghost[3]
    grid_vy = xghost[1], xvi[2], xghost[3]
    grid_vz = xghost[1], xghost[2], xvi[3]

    return grid_vx, grid_vy, grid_vz
end
