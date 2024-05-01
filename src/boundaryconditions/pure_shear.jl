function pureshear_bc!(
    stokes::JustRelax.StokesArrays, xci::NTuple{2,T}, xvi::NTuple{2,T}, εbg
) where {T}
    _T = typeof(stokes.V.Vx)
    stokes.V.Vx[:, 2:(end - 1)] .= _T([εbg * x for x in xvi[1], y in xci[2]])
    stokes.V.Vy[2:(end - 1), :] .= _T([-εbg * y for x in xci[1], y in xvi[2]])

    return nothing
end

function pureshear_bc!(
    stokes::JustRelax.StokesArrays, xci::NTuple{3,T}, xvi::NTuple{3,T}, εbg
) where {T}
    xv, yv, zv = xvi
    xc, yc, zc = xci
    _T = typeof(stokes.V.Vx)

    stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .= _T([εbg * x for x in xv, y in yc, z in zc])
    stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .= _T([εbg * y for x in xc, y in xv, z in zc])
    stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .= _T([-εbg * z for x in xc, y in xc, z in zv])

    return nothing
end
