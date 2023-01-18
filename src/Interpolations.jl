# From cell vertices to cell center

@parallel function vertex2center!(center, vertex)
    @all(center) = @av(vertex)
    return nothing
end

@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

# Velocity to cell vertices

## 2D 

function velocity2vertex!(Vx_v, Vy_v, Vx, Vy; ghost_nodes = false)
    if !ghost_nodes 
        Vx2vertex_noghost!(Vx, Vx_v)
        Vy2vertex_noghost!(Vy, Vy_v)
    else
        Vx2vertex_ghost!(Vx, Vx_v)
        Vy2vertex_ghost!(Vy, Vy_v)
    end
end

function velocity2vertex(Vx, Vy, nv_x, nv_y; ghost_nodes = false)
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
        V[i, j] = 0.5 * (Vx[i, j-1] + Vx[i, j])

    elseif j == 1
        V[i, j] = Vx[i, j]
    
    elseif j == size(Vx, 2)
        V[i, j] = Vx[i, end]
    end
    return nothing
end

@parallel_indices (i, j) function Vy2vertex_noghost!(V, Vy)
    if 1 < i < size(Vy, 1)
        V[i, j] = 0.5 * (Vy[i-1, j] + Vy[i, j])

    elseif i == 1
        V[i, j] = Vy[i, j]
    
    elseif i == size(Vy, 1)
        V[i, j] = Vy[end, j]
    end
    return nothing
end

@parallel_indices (i, j) function Vx2vertex_ghost!(V, Vx)
    if 1 < j < size(Vx, 2)
        V[i, j] = 0.5 * (Vx[i+1, j-1] + Vx[i+1, j])

    elseif i == 1
        V[i, j] = Vx[i+1, j]
    
    elseif j == size(Vx, 2)

        V[i, j] = Vx[i+1, end]
    end
    return nothing
end

@parallel_indices (i, j) function Vy2vertex_ghost!(V, Vy)
    if 1 < i < size(Vy, 1)
        V[i, j] = 0.5 * (Vy[i-1, j+1] + Vy[i, j+1])

    elseif i == 1
        V[i, j] = Vx[i, j+1]
    
    elseif i == size(Vy, 1)
        V[i, j] = Vx[end, j+1]
    end
    return nothing
end


## 3D 

function velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz; ghost_nodes = false)
    if !ghost_nodes 
        Vx2vertex_noghost!(Vx, Vx_v)
        Vy2vertex_noghost!(Vy, Vy_v)
        Vz2vertex_noghost!(Vz, Vz_v)

    else
        Vx2vertex_ghost!(Vx, Vx_v)
        Vy2vertex_ghost!(Vy, Vy_v)
        Vz2vertex_ghost!(Vz, Vz_v)
    end
end

function velocity2vertex(Vx, Vy, Vz, nv_x, nv_y, nv_z; ghost_nodes = false)
    Vx_v = @allocate(nv_x, nv_y, nv_z)
    Vy_v = @allocate(nv_x, nv_y, nv_z)
    Vz_v = @allocate(nv_x, nv_y, nv_z)

    if !ghost_nodes 
        Vx2vertex_noghost!(Vx, Vx_v)
        Vy2vertex_noghost!(Vy, Vy_v)
        Vz2vertex_noghost!(Vz, Vz_v)

    else
        Vx2vertex_ghost!(Vx, Vx_v)
        Vy2vertex_ghost!(Vy, Vy_v)
        Vz2vertex_ghost!(Vz, Vz_v)
    end
    return nothing
end

@parallel_indices (i, j, k) function Vx2vertex_noghost!(V, Vx)
    nx, ny, nz = size(Vx)
    if (1 < j < ny) && (1 < k < nz)
        V[i, j, k] = 0.25 * (Vx[i-1, j-1, k-1] + Vx[i-1, j, k-1] + Vx[i-1, j-1, k] + Vx[i-1, j, k])

    # Corners
    elseif (i,j,k) == (1,1,1)
        V[i, j, k] = Vx[i, j, k]

    elseif (i,j,k) == (1,1,nk)
        V[i, j, k] = Vx[i, j, end]

    elseif (i,j,k) == (1,ny,1)
        V[i, j, k] = Vx[i, end, k]

    elseif (i,j,k) == (nx,1,1)
        V[i, j, k] = Vx[end, j, k]

    end
    return nothing
end

@parallel_indices (i, j, k) function Vy2vertex_noghost!(V, Vy)
    if (1 < i < size(Vy, 1)) && (1 < k < size(Vy, 3))
        V[i, j, k] = 0.25 * (Vx[i-1, j-1, k-1] + Vx[i, j-1, k-1] + Vx[i-1, j-1, k] + Vx[i, j-1, k])

    # Corners
    elseif (i,j,k) == (1,1,1)
        V[i, j, k] = Vy[i, j, k]

    elseif (i,j,k) == (1,1,nk)
        V[i, j, k] = Vy[i, j, end]

    elseif (i,j,k) == (1,ny,1)
        V[i, j, k] = Vy[i, end, k]

    elseif (i,j,k) == (nx,1,1)
        V[i, j, k] = Vy[end, j, k]


    # Planes
    elseif j == 1 && (1 < i < nx) && (1 < k < nz) # xz front plane
        V[i, j, k] = Vy[i, j, k]

    elseif j == ny && (1 < i < nx) && (1 < k < nz) # xz back plane
        V[i, j, k] = Vy[i, j, k]

    elseif i == 1 && (1 < j < ny) && (1 < k < nz) # yz left plane
        V[i, j, k] = Vy[i, j, k]

    elseif i == nx && (1 < j < ny) && (1 < k < nz) # yz right plane
        V[i, j, k] = Vy[i, j, k]

    elseif k == 1 && (1 < i < nx) && (1 < j < ny) # xy bottom plane
        V[i, j, k] = Vy[i, j, k]

    elseif k == nk && (1 < i < nx) && (1 < j < ny) # xy top plane
        V[i, j, k] = Vy[i, j, k]

    end
    return nothing
end

@parallel_indices (i, j, k) function Vz2vertex_noghost!(V, Vz)
    if (1 < i < size(Vz, 1)) && (1 < j < size(Vz, 2))
        V[i, j, k] = 0.25 * (Vx[i-1, j-1, k-1] + Vx[i-1, j, k-1] + Vx[i, j-1, k-1] + Vx[i, j, k-1])

    # Corners
    elseif (i,j,k) == (1,1,1)
        V[i, j, k] = Vz[i, j, k]

    elseif (i,j,k) == (1,1,nk)
        V[i, j, k] = Vz[i, j, end]

    elseif (i,j,k) == (1,ny,1)
        V[i, j, k] = Vz[i, end, k]

    elseif (i,j,k) == (nx,1,1)
        V[i, j, k] = Vz[end, j, k]
    end
    return nothing
end


@parallel_indices (i, j, k) function Vx2vertex_ghost!(V, Vx)
    if (1 < j < size(Vx, 2)) && (1 < k < size(Vx, 3))
        V[i, j, k] = 0.25 * (Vx[i, j-1, k-1] + Vx[i, j, k-1] + Vx[i-1, j-1, k] + Vx[i, j, k])

    # elseif i == 1
    #     V[i, j, k] = Vx[i, j, k]
    
    # else
    #     V[i, j, k] = Vx[i, j-1, k]
    end
    return nothing
end

@parallel_indices (i, j, k) function Vy2vertex_ghost!(V, Vy)
    if (1 < i < size(Vy, 1)) && (1 < k < size(Vy, 3))
        V[i, j, k] = 0.25 * (Vx[i-1, j, k-1] + Vx[i, j, k-1] + Vx[i-1, j-1, k] + Vx[i, j, k])

    # elseif i == 1
    #     V[i, j, k] = Vx[i, j, k]
    
    # else
    #     V[i, j, k] = Vx[i, j-1, k]
    end
    return nothing
end

@parallel_indices (i, j, k) function Vz2vertex_ghost!(V, Vz)
    if (1 < i < size(Vz, 1)) && (1 < j < size(Vz, 2))
        V[i, j, k] = 0.25 * (Vx[i-1, j-1, k] + Vx[i-1, j, k] + Vx[i, j-1, k-1] + Vx[i, j, k])

    # elseif i == 1
    #     V[i, j, k] = Vx[i, j, k]
    
    # else
    #     V[i, j, k] = Vx[i, j-1, k]
    end
    return nothing
end
