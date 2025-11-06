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

    iW = clamp(i - 1, 1, nx)
    iE = clamp(i + 1, 1, nx)
    jS = clamp(j - 1, 1, ny)
    jN = clamp(j, 1, ny)

    # OPTION 1
    ρg_L = 0.25 * (ρg[iW, jS] + ρg[i, jS] + ρg[iW, jN] + ρg[i, jN])
    ρg_R = 0.25 * (ρg[iE, jS] + ρg[i, jS] + ρg[iE, jN] + ρg[i, jN])

    Vx_on_Vy[i + 1, j] =
        (0.25 * (Vx[i, j] + Vx[i + 1, j] + Vx[i, j + 1] + Vx[i + 1, j + 1])) *
        (ρg_R - ρg_L) *
        _dx

    return nothing
end

@parallel_indices (i, j) function interp_Vx∂ρ∂x_on_Vy!(Vx_on_Vy, Vx, ρg, ϕ, _dx)
    nx, ny = size(ρg)

    iW = clamp(i - 1, 1, nx)
    iE = clamp(i + 1, 1, nx)
    jS = clamp(j - 1, 1, ny)
    jN = clamp(j, 1, ny)

    # OPTION 1
    ρg_L =
        0.25 * (
        ρg[iW, jS] * ϕ.center[iW, jS] +
            ρg[i, jS] * ϕ.center[i, jS] +
            ρg[iW, jN] * ϕ.center[iW, jN] +
            ρg[i, jN] * ϕ.center[i, jN]
    )
    ρg_R =
        0.25 * (
        ρg[iE, jS] * ϕ.center[iE, jS] +
            ρg[i, jS] * ϕ.center[i, jS] +
            ρg[iE, jN] * ϕ.center[iE, jN] +
            ρg[i, jN] * ϕ.center[i, jN]
    )

    Vx_on_Vy[i + 1, j] =
        (0.25 * (Vx[i, j] + Vx[i + 1, j] + Vx[i, j + 1] + Vx[i + 1, j + 1])) *
        (ρg_R - ρg_L) *
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
    ) where {T <: AbstractArray{_T, 2} where {_T <: Real}}
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
    ) where {T <: AbstractArray{_T, 3} where {_T <: Real}}
    @inline av_T() = _av(T_vertex, i, j, k)

    T_center[i, j, k] = av_T()

    return nothing
end

"""
    vertex2center!(center, vertex)

Interpolates the values at the `vertex` onto `center` points.
"""

function vertex2center!(center, vertex)
    @parallel vertex2center_kernel!(center, vertex)
    return nothing
end

@parallel function vertex2center_kernel!(center, vertex)
    @all(center) = @av(vertex)
    return nothing
end

"""
    center2vertex!(vertex, center)

Interpolates the values at the `center` onto `vertex` points.
"""

function center2vertex!(vertex, center)
    @parallel center2vertex_kernel!(vertex, center)
    @views vertex[1, :] .= vertex[2, :]
    @views vertex[end, :] .= vertex[end - 1, :]
    @views vertex[:, 1] .= vertex[:, 2]
    @views vertex[:, end] .= vertex[:, end - 1]

    return nothing
end

@parallel function center2vertex_kernel!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

function center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
    @parallel center2vertex_kernel!(
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

# 3D

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
    @assert size(Vx_v) == size(Vy_v) == size(Vz_v)
    # interpolate to cell vertices
    @parallel (@idx size(Vx_v)) _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    return nothing
end

@parallel_indices (i, j, k) function _velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    Vx_v[i, j, k] =
        0.25 * (Vx[i, j, k] + Vx[i, j + 1, k] + Vx[i, j, k + 1] + Vx[i, j + 1, k + 1])
    Vy_v[i, j, k] =
        0.25 * (Vy[i, j, k] + Vy[i + 1, j, k] + Vy[i, j, k + 1] + Vy[i + 1, j, k + 1])
    Vz_v[i, j, k] =
        0.25 * (Vz[i, j, k] + Vz[i, j + 1, k] + Vz[i + 1, j, k] + Vz[i + 1, j + 1, k])
    return nothing
end
# 2D

"""
    velocity2vertex(Vx, Vy)

Interpolate the velocity field `Vx`, `Vy` from a staggered grid with ghost nodes
onto the grid vertices.
"""

function velocity2vertex!(Vx_v, Vy_v, Vx, Vy)
    @assert size(Vx_v) == size(Vy_v)
    # interpolate to cell vertices
    @parallel (@idx size(Vx_v)) _velocity2vertex!(Vx_v, Vy_v, Vx, Vy)
    return nothing
end

@parallel_indices (i, j, k) function _velocity2vertex!(Vx_v, Vy_v, Vx, Vy)
    Vx_v[i, j, k] = (Vx[i, j] + Vx[i, j + 1]) / 2
    Vy_v[i, j, k] = (Vy[i, j] + Vy[i + 1, j]) / 2
    return nothing
end

"""
    velocity2center(Vx_c, Vy_c, Vz_c, Vx, Vy, Vz)

Interpolate the velocity field `Vx`, `Vy`, `Vz` from a staggered grid with ghost nodes
onto the grid centers.
"""

function velocity2center!(Vx_c, Vy_c, Vz_c, Vx, Vy, Vz)
    @assert size(Vx_c) == size(Vy_c) == size(Vz_c)
    # interpolate to cell vertices
    @parallel (@idx size(Vx_c)) _velocity2center!(Vx_c, Vy_c, Vz_c, Vx, Vy, Vz)
    return nothing
end

@parallel_indices (i, j, k) function _velocity2center!(Vx_c, Vy_c, Vz_c, Vx, Vy, Vz)
    Vx_c[i, j, k] = (Vx[i, j + 1, k + 1] + Vx[i + 1, j + 1, k + 1]) / 2
    Vy_c[i, j, k] = (Vy[i + 1, j, k + 1] + Vy[i + 1, j + 1, k + 1]) / 2
    Vz_c[i, j, k] = (Vz[i + 1, j + 1, k] + Vz[i + 1, j + 1, k + 1]) / 2
    return nothing
end

"""
    velocity2center(Vx_c, Vy_c, Vx, Vy)

Interpolate the velocity field `Vx`, `Vy` from a staggered grid with ghost nodes
onto the grid centers.
"""

function velocity2center!(Vx_c, Vy_c, Vx, Vy)
    @assert size(Vx_c) == size(Vy_c)
    # interpolate to cell vertices
    @parallel (@idx size(Vx_c)) _velocity2center!(Vx_c, Vy_c, Vx, Vy)
    return nothing
end

@parallel_indices (i, j, k) function _velocity2center!(Vx_c, Vy_c, Vx, Vy)
    Vx_c[i, j, k] = (Vx[i, j + 1] + Vx[i + 1, j + 1]) / 2
    Vy_c[i, j, k] = (Vy[i + 1, j] + Vy[i + 1, j + 1]) / 2
    return nothing
end

function shear2center!(A::JustRelax.SymmetricTensor)
    return shear2center!(backend(A), A)
end

function shear2center!(::CPUBackendTrait, A::JustRelax.SymmetricTensor)
    _shear2center!(A)
    return nothing
end

function _shear2center!(A::JustRelax.SymmetricTensor)
    @parallel (@idx size(A.xy_c)) shear2center_kernel!(@shear_center(A), @shear(A))
    return nothing
end

# 2D
@parallel_indices (i, j) function shear2center_kernel!(
        xy_c::T, xy::T
    ) where {T <: AbstractArray{_T, 2} where {_T <: Real}}
    xy_c[i, j] = 0.25 * (xy[i, j] + xy[i + 1, j] + xy[i, j + 1] + xy[i + 1, j + 1])
    return nothing
end

# 3D
@parallel_indices (i, j, k) function shear2center_kernel!(
        center::NTuple{3, T}, shear::NTuple{3, T}
    ) where {T <: AbstractArray{_T, 3} where {_T <: Real}}
    yz_c, xz_c, xy_c = center
    yz, xz, xy = shear
    yz_c[i, j, k] = 0.25 * (yz[i, j, k] + yz[i, j + 1, k] + yz[i, j, k + 1] + yz[i, j + 1, k + 1])
    xz_c[i, j, k] = 0.25 * (xz[i, j, k] + xz[i + 1, j, k] + xz[i, j, k + 1] + xz[i + 1, j, k + 1])
    xy_c[i, j, k] = 0.25 * (xy[i, j, k] + xy[i + 1, j, k] + xy[i, j + 1, k] + xy[i + 1, j + 1, k])
    return nothing
end
