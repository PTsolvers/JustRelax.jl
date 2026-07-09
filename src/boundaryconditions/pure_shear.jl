"""
    pureshear_bc!(stokes, xci, xvi, εbg, backend)

Prescribe a pure-shear velocity field on the boundaries of `stokes.V` for a background
strain rate `εbg`. The velocity increases linearly with position, using the cell-center and
vertex coordinates `xci` and `xvi`, so the imposed rates are `εbg` along the horizontal
directions and `-εbg` along the vertical, keeping the flow incompressible. Works in 2D and
3D depending on the length of the coordinate tuples.
"""
function pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{2}, xvi::NTuple{2}, εbg, backend
    )
    stokes.V.Vx[:, 2:(end - 1)] .= PTArray(backend)([εbg * x for x in xvi[1], y in xci[2]])
    stokes.V.Vy[2:(end - 1), :] .= PTArray(backend)([-εbg * y for x in xci[1], y in xvi[2]])

    return nothing
end

function pureshear_bc!(
        stokes::JustRelax.StokesArrays, xci::NTuple{3}, xvi::NTuple{3}, εbg, backend
    )
    xv, yv, zv = xvi
    xc, yc, zc = xci

    stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .= PTArray(backend)(
        [
            εbg * x for x in xv, y in yc, z in zc
        ]
    )
    stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .= PTArray(backend)(
        [
            εbg * y for x in xc, y in xv, z in zc
        ]
    )
    stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .= PTArray(backend)(
        [
            -εbg * z for x in xc, y in xc, z in zv
        ]
    )

    return nothing
end
