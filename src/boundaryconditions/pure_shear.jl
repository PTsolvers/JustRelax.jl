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

@inline _pureshear_coordinate(::Type{CPUBackend}, x) = x
@inline _pureshear_coordinate(backend, x) = PTArray(backend)(x)
@inline _pureshear_coordinate(::Array, x) = x
function _pureshear_coordinate(A::AbstractArray, x)
    x_backend = similar(A, eltype(x), (length(x),))
    copyto!(x_backend, x)
    return x_backend
end

"""
    pureshear_bc!(stokes, xci, xvi, εbg)
    pureshear_bc!(stokes, xci, xvi, εbg, backend)

Initialize a pure-shear background velocity field on the staggered grids.
`xci` contains cell-center coordinates and `xvi` contains velocity-grid
coordinates. In 2D, the kernels set
`Vx = εbg*x` and `Vy = -εbg*y`; in 3D they additionally set
`Vz = -εbg*z`. Ghost layers are left untouched so that subsequent flow
boundary-condition and halo updates can set them consistently.

The preferred form infers the coordinate-array backend from `stokes`. The
five-argument form remains available for compatibility and explicitly selects
the backend. All field updates are performed by ParallelStencil kernels.
"""
function pureshear_bc!(stokes::JustRelax.StokesArrays, xci::NTuple{N}, xvi::NTuple{N}, εbg) where {N}
    return _pureshear_bc!(stokes, xci, xvi, εbg, stokes.V.Vx)
end

function pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, εbg, backend
    )
    xv, yv = (_pureshear_coordinate(backend, x) for x in xvi)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2)) _pureshear_x_2d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2))) _pureshear_y_2d!(Vy, yv, εbg)
    return nothing
end

function _pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, εbg, coordinate_array
    )
    xv, yv = (_pureshear_coordinate(coordinate_array, x) for x in xvi)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2)) _pureshear_x_2d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2))) _pureshear_y_2d!(Vy, yv, εbg)
    return nothing
end

function pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, εbg, backend
    )
    xv, yv, zv = (_pureshear_coordinate(backend, x) for x in xvi)
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2, size(Vx, 3) - 2)) _pureshear_x_3d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2), size(Vy, 3) - 2)) _pureshear_y_3d!(Vy, yv, εbg)
    @parallel (@idx (size(Vz, 1) - 2, size(Vz, 2) - 2, size(Vz, 3))) _pureshear_z_3d!(Vz, zv, εbg)
    return nothing
end

function _pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, εbg, coordinate_array
    )
    xv, yv, zv = (_pureshear_coordinate(coordinate_array, x) for x in xvi)
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2, size(Vx, 3) - 2)) _pureshear_x_3d!(Vx, xv, εbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2), size(Vy, 3) - 2)) _pureshear_y_3d!(Vy, yv, εbg)
    @parallel (@idx (size(Vz, 1) - 2, size(Vz, 2) - 2, size(Vz, 3))) _pureshear_z_3d!(Vz, zv, εbg)
    return nothing
end
