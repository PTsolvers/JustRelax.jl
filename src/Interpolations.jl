# From cell vertices to cell center

@parallel function vertex2center!(center, vertex)
    @all(center) = @av(vertex)
    return nothing
end

@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

@parallel_indices (i, j, k) function center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
    
    nx, ny, nz = size(center_xz)
    
    Base.@propagate_inbounds @inline function clamp_idx(i, j, k)
        i = clamp(i, 1, nx)
        j = clamp(j, 1, ny)
        k = clamp(k, 1, nz)
        i, j, k
    end

    @inbounds begin
        # if all( (i,j,k) .≤ size(vertex_yz))
        if i ≤ size(vertex_yz, 1) && (1 < j < size(vertex_yz, 2)) && (1 < k < size(vertex_yz, 3))
            vertex_yz[i, j, k] = 0.25 * (
                center_yz[clamp_idx(i, j-1, k-1)...] +
                center_yz[clamp_idx(i, j  , k-1)...] +
                center_yz[clamp_idx(i, j-1, k  )...] +
                center_yz[clamp_idx(i, j  , k  )...]
            )
        end
        # if all( (i,j,k) .≤ size(vertex_xz))
        if (1 < i < size(vertex_xz, 1)) && j ≤ size(vertex_xz, 2) && (1 < k < size(vertex_xz, 3))
            vertex_xz[i, j, k] = 0.25 * (
                center_xz[clamp_idx(i-1, j, k-1)...] +
                center_xz[clamp_idx(i  , j, k-1)...] +
                center_xz[clamp_idx(i-1, j, k  )...] +
                center_xz[clamp_idx(i  , j, k  )...]
            )
        end
        # if all( (i,j,k) .≤ size(vertex_xy))
        if (1 < i < size(vertex_xy, 1)) && (1 < j < size(vertex_xy, 2)) && k ≤ size(vertex_xy, 3)
            vertex_xy[i, j, k] = 0.25 * (
                center_xy[clamp_idx(i-1, j-1, k)...] +
                center_xy[clamp_idx(i  , j-1, k)...] +
                center_xy[clamp_idx(i-1, j  , k)...] +
                center_xy[clamp_idx(i  , j  , k)...]
            )
        end
    end
    
    return nothing
end

# Velocity to cell vertices

## 2D 

function velocity2vertex!(Vx_v, Vy_v, Vx, Vy; ghost_nodes=false)
    if !ghost_nodes
        Vx2vertex_noghost!(Vx, Vx_v)
        Vy2vertex_noghost!(Vy, Vy_v)
    else
        Vx2vertex_ghost!(Vx, Vx_v)
        Vy2vertex_ghost!(Vy, Vy_v)
    end
end

function velocity2vertex(Vx, Vy, nv_x, nv_y; ghost_nodes=false)
    Vx_v = @allocate(nv_x, nv_y)
    Vy_v = @allocate(nv_x, nv_y)

    if !ghost_nodes
        Vx2vertex_noghost!(Vx, Vx_v)
        Vy2vertex_noghost!(Vy, Vy_v)
    else
        Vx2vertex_ghost!(Vx, Vx_v)
        Vy2vertex_ghost!(Vy, Vy_v)
    end
end

@parallel_indices (i, j) function Vx2vertex_noghost!(V, Vx)
    if 1 < j < size(Vx, 2)
        V[i, j] = 0.5 * (Vx[i, j - 1] + Vx[i, j])

    elseif j == 1
        V[i, j] = Vx[i, j]

    elseif j == size(Vx, 2)
        V[i, j] = Vx[i, end]
    end
    return nothing
end

@parallel_indices (i, j) function Vy2vertex_noghost!(V, Vy)
    if 1 < i < size(Vy, 1)
        V[i, j] = 0.5 * (Vy[i - 1, j] + Vy[i, j])

    elseif i == 1
        V[i, j] = Vy[i, j]

    elseif i == size(Vy, 1)
        V[i, j] = Vy[end, j]
    end
    return nothing
end

@parallel_indices (i, j) function Vx2vertex_ghost!(V, Vx)
    if 1 < j < size(Vx, 2)
        V[i, j] = 0.5 * (Vx[i + 1, j - 1] + Vx[i + 1, j])

    elseif i == 1
        V[i, j] = Vx[i + 1, j]

    elseif j == size(Vx, 2)
        V[i, j] = Vx[i + 1, end]
    end
    return nothing
end

@parallel_indices (i, j) function Vy2vertex_ghost!(V, Vy)
    if 1 < i < size(Vy, 1)
        V[i, j] = 0.5 * (Vy[i - 1, j + 1] + Vy[i, j + 1])

    elseif i == 1
        V[i, j] = Vx[i, j + 1]

    elseif i == size(Vy, 1)
        V[i, j] = Vx[end, j + 1]
    end
    return nothing
end

## 3D 

"""
    velocity2vertex(Vx, Vy, Vz)

Interpolate the velocity field `Vx`, `Vy`, `Vz` from a staggered grid with ghost nodes 
onto the grid vertices.
"""
function velocity2vertex(Vx, Vy, Vz)
    # infer size of grid
    nx, ny, nz = size(Vx)
    nv_x, nv_y, nv_z = nx - 1, ny - 2, nz - 2
    # allocate output arrays
    Vx_v = @zeros(nv_x, nv_y, nv_z)
    Vy_v = @zeros(nv_x, nv_y, nv_z)
    Vz_v = @zeros(nv_x, nv_y, nv_z)
    # interpolate to cell vertices
    @parallel (@idx nv_x nv_y nv_z) _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)

    return Vx_v, Vy_v, Vz_v
end

"""
    velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)

In-place interpolation of the velocity field `Vx`, `Vy`, `Vz` from a staggered grid with ghost nodes 
onto the pre-allocated `Vx_d`, `Vy_d`, `Vz_d` 3D arrays located at the grid vertices.
"""
function velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    # infer size of grid
    nx, ny, nz = size(Vx)
    nv_x, nv_y, nv_z = nx - 1, ny - 2, nz - 2
    # interpolate to cell vertices
    @parallel (@idx nv_x nv_y nv_z) _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)

    return nothing 
end

@parallel_indices (i, j, k) function _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    @inbounds begin
        Vx_v[i, j, k] = 0.25 * (
            Vx[i, j    , k    ] +
            Vx[i, j + 1, k    ] +
            Vx[i, j    , k + 1] +
            Vx[i, j + 1, k + 1]
        )
        Vy_v[i, j, k] = 0.25 * (
            Vy[i    , j, k    ] +
            Vy[i + 1, j, k    ] +
            Vy[i    , j, k + 1] +
            Vy[i + 1, j, k + 1]
        )
        Vz_v[i, j, k] = 0.25 * (
            Vz[i    , j    , k] +
            Vz[i    , j + 1, k] +
            Vz[i + 1, j    , k] +
            Vz[i + 1, j + 1, k]
        )

    end
    return nothing
end