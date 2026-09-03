@parallel_indices (i, j) function _pureshear_x_2d!(Vx, xv, εbg)
    @inbounds Vx[i, j + 1] = εbg * xv[i]
    return nothing
end

@parallel_indices (i, j) function _pureshear_y_2d!(Vy, yv, εbg)
    @inbounds Vy[i + 1, j] = -εbg * yv[j]
    return nothing
end

@parallel_indices (i, j, k) function _pureshear_x_3d!(Vx, xv, εbg)
    @inbounds Vx[i, j + 1, k + 1] = εbg * xv[i]
    return nothing
end

@parallel_indices (i, j, k) function _pureshear_y_3d!(Vy, yv, εbg)
    @inbounds Vy[i + 1, j, k + 1] = εbg * yv[j]
    return nothing
end

@parallel_indices (i, j, k) function _pureshear_z_3d!(Vz, zv, εbg)
    @inbounds Vz[i + 1, j + 1, k] = -εbg * zv[k]
    return nothing
end

"""
    _bc_coordinate(A, x)

Materialize coordinate vector `x` on the same device as array `A`, so that
background-field kernels can index it. `collect` is applied first because a
lazy range cannot be transferred to a device array directly.
"""
@inline _bc_coordinate(::Array, x) = collect(x)
function _bc_coordinate(A::AbstractArray, x)
    x_backend = similar(A, eltype(x), (length(x),))
    copyto!(x_backend, collect(x))
    return x_backend
end

"""
    pureshear_bc!(stokes, xci, xvi, εbg)
    pureshear_bc!(stokes, xci, xvi, εbg, backend)

Initialize a pure-shear background velocity field on the staggered grids.
`xci` contains cell-center coordinates and `xvi` contains velocity-grid
coordinates; each component is built from the vertex coordinates of its own
direction. In 2D the kernels set `Vx = εbg*x` and `Vy = -εbg*y`. In 3D they set
`Vx = εbg*x`, `Vy = εbg*y`, and `Vz = -εbg*z`. Ghost layers are left untouched
so that subsequent flow boundary-condition and halo updates can set them
consistently.

All field updates are performed by ParallelStencil kernels on the backend of
`stokes`. The five-argument form remains available for compatibility; its
`backend` argument is redundant, as the backend is inferred from `stokes`.
"""
function pureshear_bc!(stokes::JustRelax.StokesArrays, xci, xvi, εbg)
    return pureshear_bc!(backend(stokes), stokes, xci, xvi, εbg)
end

function pureshear_bc!(stokes::JustRelax.StokesArrays, xci, xvi, εbg, backend)
    return pureshear_bc!(stokes, xci, xvi, εbg)
end

function pureshear_bc!(::CPUBackendTrait, stokes::JustRelax.StokesArrays, xci, xvi, εbg)
    return _pureshear_bc!(stokes, xci, xvi, εbg)
end

function _pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, εbg
    )
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    xv, yv = ntuple(i -> _bc_coordinate(Vx, xvi[i]), Val(2))

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2)) _pureshear_x_2d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2))) _pureshear_y_2d!(Vy, yv, εbg)
    return nothing
end

function _pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, εbg
    )
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
    xv, yv, zv = ntuple(i -> _bc_coordinate(Vx, xvi[i]), Val(3))

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2, size(Vx, 3) - 2)) _pureshear_x_3d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2), size(Vy, 3) - 2)) _pureshear_y_3d!(Vy, yv, εbg)
    @parallel (@idx (size(Vz, 1) - 2, size(Vz, 2) - 2, size(Vz, 3))) _pureshear_z_3d!(Vz, zv, εbg)
    return nothing
end
