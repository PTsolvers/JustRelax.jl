"""
    interp_Vx_on_Vy!(Vx_on_Vy, Vx)

Interpolates the values of `Vx` onto the grid points of `Vy`.

# Arguments
- `Vx_on_Vy::AbstractArray`: `Vx` at `Vy` grid points.
- `Vx::AbstractArray`: `Vx` at its staggered grid points.
"""
@parallel_indices (i, j) function interp_Vx_on_Vy!(Vx_on_Vy, Vx)
    Vx_on_Vy[i + 1, j] = 0.25 * (Vx[i, j] + Vx[i + 1, j] + Vx[i, j + 1] + Vx[i + 1, j + 1])
    return nothing
end

@parallel_indices (i, j) function interp_Vx∂ρ∂x_on_Vy!(Vx_on_Vy, Vx, ρg, _dx)
    nx, ny = size(ρg)
    ii = clamp(i, 1, nx)
    ii1 = clamp(i + 1, 1, nx)
    jj = clamp(j, 1, ny)
    Vx_on_Vy[i + 1, j] =
        (0.25 * (Vx[i, j] + Vx[i + 1, j] + Vx[i, j + 1] + Vx[i + 1, j + 1])) *
        (ρg[ii1, jj] - ρg[ii, jj]) *
        _dx
    return nothing
end

# From cell vertices to cell center

temperature2center!(thermal) = temperature2center!(backend(thermal), thermal)
function temperature2center!(::CPUBackendTrait, thermal::JustRelax.ThermalArrays)
    return _temperature2center!(thermal)
end

function _temperature2center!(thermal::JustRelax.ThermalArrays)
    @parallel (@idx size(thermal.Tc)...) temperature2center_kernel!(thermal.Tc, thermal.T)
    return nothing
end

@parallel_indices (i, j) function temperature2center_kernel!(
    T_center::T, T_vertex::T
) where {T<:AbstractArray{_T,2} where {_T<:Real}}
    T_center[i, j] =
        (
            T_vertex[i + 1, j] +
            T_vertex[i + 2, j] +
            T_vertex[i + 1, j + 1] +
            T_vertex[i + 2, j + 1]
        ) * 0.25
    return nothing
end

@parallel_indices (i, j, k) function temperature2center_kernel!(
    T_center::T, T_vertex::T
) where {T<:AbstractArray{_T,3} where {_T<:Real}}
    @inline av_T() = _av(T_vertex, i, j, k)

    T_center[i, j, k] = av_T()

    return nothing
end

function vertex2center!(center, vertex)
    @parallel vertex2center_kernel!(center, vertex)
    return nothing
end

@parallel function vertex2center_kernel!(center, vertex)
    @all(center) = @av(vertex)
    return nothing
end

function center2vertex!(vertex, center)
    @parallel center2vertex_kernel!(vertex, center)
    return nothing
end

@parallel function center2vertex_kernel!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

function center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
    ni = size(center_yz)
    
    @parallel (@idx ni.+1) center2vertex_kernel!(
        vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy
    )
    return nothing
end

@parallel_indices (i, j, k) function center2vertex_kernel!(
    vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy
)
    i1, j1, k1 = (i, j, k) .+ 1
    nx, ny, nz = size(center_yz)
    if i ≤ nx && j1 ≤ ny && k1 ≤ nz
        vertex_yz[i, j1, k1] =
            0.25 * (
                center_yz[i, j, k] +
                center_yz[i, j1, k] +
                center_yz[i, j, k1] +
                center_yz[i, j1, k1]
            )
    end
    if i1 ≤ nx && j ≤ ny && k1 ≤ nz
        vertex_xz[i1, j, k1] =
            0.25 * (
                center_xz[i, j, k] +
                center_xz[i1, j, k] +
                center_xz[i, j, k1] +
                center_xz[i1, j, k1]
            )
    end
    if i1 ≤ nx && j1 ≤ ny && k ≤ nz
        vertex_xy[i1, j1, k] =
            0.25 * (
                center_xy[i, j, k] +
                center_xy[i1, j, k] +
                center_xy[i, j1, k] +
                center_xy[i1, j1, k]
            )
    end
    return nothing
end

# Velocity to cell vertices

## 2D 

function velocity2vertex!(Vx_v, Vy_v, Vx, Vy; ghost_nodes=true)
    ni = size(Vx_v)
    if !ghost_nodes
        @parallel (@idx ni) Vx2vertex_noghost!(Vx_v, Vx)
        @parallel (@idx ni) Vy2vertex_noghost!(Vy_v, Vy)
    else
        @parallel (@idx ni) Vx2vertex_LinP!(Vx_v, Vx)
        @parallel (@idx ni) Vy2vertex_LinP!(Vy_v, Vy)
    end
end

function velocity2vertex(Vx, Vy, nv_x, nv_y; ghost_nodes=true)
    Vx_v = @allocate(nv_x, nv_y)
    Vy_v = @allocate(nv_x, nv_y)

    if !ghost_nodes
        @parallel (@idx ni) Vx2vertex_noghost!(Vx_v, Vx)
        @parallel (@idx ni) Vy2vertex_noghost!(Vy_v, Vy)
    else
        @parallel (@idx ni) Vx2vertex_LinP!(Vx_v, Vx)
        @parallel (@idx ni) Vy2vertex_LinP!(Vy_v, Vy)
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
    @inline av(A) = _av_ya(A, i, j)

    V[i, j] = av(Vx)
    return nothing
end

@parallel_indices (i, j) function Vy2vertex_ghost!(V, Vy)
    @inline av(A) = _av_xa(A, i, j)
    V[i, j] = av(Vy)

    return nothing
end

@parallel_indices (i, j) function Vx2vertex_LinP!(V, Vx)
    @inline av(A, B) = (A + B) * 0.5

    nx, ny = size(Vx)

    iSW, jSW = clamp(i - 1, 1, nx), clamp(j, 1, ny)
    iS, jS = clamp(i, 1, nx), clamp(j, 1, ny)
    iSE, jSE = clamp(i + 1, 1, nx), clamp(j, 1, ny)

    iNE, jNE = clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)
    iN, jN = clamp(i, 1, nx), clamp(j + 1, 1, ny)
    iNW, jNW = clamp(i - 1, 1, nx), clamp(j + 1, 1, ny)

    V_SW = av(Vx[iSW, jSW], Vx[iS, jS])
    V_SE = av(Vx[iS, jS], Vx[iSE, jSE])
    V_NW = av(Vx[iNW, jNW], Vx[iN, jN])
    V_NE = av(Vx[iN, jN], Vx[iNE, jNE])

    V[i, j] = 0.25 * (V_SW + V_SE + V_NW + V_NE)

    return nothing
end

@parallel_indices (i, j) function Vy2vertex_LinP!(V, Vy)
    @inline av(A, B) = (A + B) * 0.5

    nx, ny = size(Vy)

    iSW, jSW = clamp(i, 1, nx), clamp(j - 1, 1, ny)
    iW, jW = clamp(i, 1, nx), clamp(j, 1, ny)
    iSE, jSE = clamp(i, 1, nx), clamp(j + 1, 1, ny)

    iNE, jNE = clamp(i + 1, 1, nx), clamp(j - 1, 1, ny)
    iE, jE = clamp(i + 1, 1, nx), clamp(j, 1, ny)
    iNW, jNW = clamp(i + 1, 1, nx), clamp(j + 1, 1, ny)

    V_SW = av(Vy[iSW, jSW], Vy[iW, jW])
    V_SE = av(Vy[iW, jW], Vy[iSE, jSE])
    V_NW = av(Vy[iNW, jNW], Vy[iE, jE])
    V_NE = av(Vy[iE, jE], Vy[iNE, jNE])

    V[i, j] = 0.25 * (V_SW + V_SE + V_NW + V_NE)

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
    n = max(nx, ny, nz)
    nv_x, nv_y, nv_z = nx - 1, ny - 2, nz - 2
    # interpolate to cell vertices
    @parallel (@idx n, n, n) _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)

    return nothing
end

@parallel_indices (i, j, k) function _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    @inbounds begin
        if all((i, j, k) .≤ size(Vx))
            Vx_v[i, j, k] =
                0.25 *
                (Vx[i, j, k] + Vx[i, j + 1, k] + Vx[i, j, k + 1] + Vx[i, j + 1, k + 1])
        end
        if all((i, j, k) .≤ size(Vy))
            Vy_v[i, j, k] =
                0.25 *
                (Vy[i, j, k] + Vy[i + 1, j, k] + Vy[i, j, k + 1] + Vy[i + 1, j, k + 1])
        end
        if all((i, j, k) .≤ size(Vz))
            Vz_v[i, j, k] =
                0.25 *
                (Vz[i, j, k] + Vz[i, j + 1, k] + Vz[i + 1, j, k] + Vz[i + 1, j + 1, k])
        end
    end
    return nothing
end
