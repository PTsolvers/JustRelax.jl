@parallel_indices (i, j) function _simpleshear_x_2d!(Vx, yc, γbg)
    @inbounds Vx[i, j + 1] = γbg * yc[j]
    return nothing
end

@parallel_indices (i, j) function _simpleshear_y_2d!(Vy)
    @inbounds Vy[i + 1, j] = zero(eltype(Vy))
    return nothing
end

@parallel_indices (i, j, k) function _simpleshear_x_3d!(Vx, yc, γbg)
    @inbounds Vx[i, j + 1, k + 1] = γbg * yc[j]
    return nothing
end

@parallel_indices (i, j, k) function _simpleshear_y_3d!(Vy)
    @inbounds Vy[i + 1, j, k + 1] = zero(eltype(Vy))
    return nothing
end

@parallel_indices (i, j, k) function _simpleshear_z_3d!(Vz)
    @inbounds Vz[i + 1, j + 1, k] = zero(eltype(Vz))
    return nothing
end

"""
    simpleshear_bc!(stokes, xci, xvi, γbg)
    simpleshear_bc!(stokes, xci, xvi, γbg, backend)

Initialize an xy simple-shear background velocity field on the staggered
grids. The imposed field is `Vx = γbg * y`; the other velocity components are
set to zero. `xci` contains cell-center coordinates and `xvi` contains
velocity-grid coordinates. Ghost layers are left untouched.
"""
function simpleshear_bc!(stokes::JustRelax.StokesArrays, xci::NTuple{N}, xvi::NTuple{N}, γbg) where {N}
    return _simpleshear_bc!(stokes, xci, xvi, γbg, stokes.V.Vx)
end

function simpleshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, γbg, backend
    )
    yc = _pureshear_coordinate(backend, xci[2])
    Vx, Vy = stokes.V.Vx, stokes.V.Vy

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2)) _simpleshear_x_2d!(Vx, yc, γbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2))) _simpleshear_y_2d!(Vy)
    return nothing
end

function _simpleshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, γbg, coordinate_array
    )
    return simpleshear_bc!(stokes, xci, xvi, γbg, coordinate_array)
end

function simpleshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, γbg, backend
    )
    yc = _pureshear_coordinate(backend, xci[2])
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz

    @parallel (@idx (size(Vx, 1), size(Vx, 2) - 2, size(Vx, 3) - 2)) _simpleshear_x_3d!(Vx, yc, γbg)
    @parallel (@idx (size(Vy, 1) - 2, size(Vy, 2), size(Vy, 3) - 2)) _simpleshear_y_3d!(Vy)
    @parallel (@idx (size(Vz, 1) - 2, size(Vz, 2) - 2, size(Vz, 3))) _simpleshear_z_3d!(Vz)
    return nothing
end

function _simpleshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, γbg, coordinate_array
    )
    return simpleshear_bc!(stokes, xci, xvi, γbg, coordinate_array)
end
