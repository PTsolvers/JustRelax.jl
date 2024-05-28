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
    _T = typeof(stokes.V.Vx)

    stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .= PTArray(backend)([
        εbg * x for x in xv, y in yc, z in zc
    ])
    stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .= PTArray(backend)([
        εbg * y for x in xc, y in xv, z in zc
    ])
    stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .= PTArray(backend)([
        -εbg * z for x in xc, y in xc, z in zv
    ])

    return nothing
end
