function analytical_velocity(xci, xvi)
    nvi = length.(xvi)
    nci = length.(xci)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx = PTArray(backend)(zeros(nvi[1], nci[2], nci[3]))
    Vy = PTArray(backend)(zeros(nci[1], nvi[2], nci[3]))
    Vz = PTArray(backend)(zeros(nci[1], nci[2], nvi[3]))

    _velocity_x(x, y, z) = x + x^2 + x * y + x^3 * y
    _velocity_y(x, y, z) = y + x * y + y^2 + x^2 * y^2
    _velocity_z(x, y, z) = -2z - 3x * z - 3y * z - 5x^2 * y * z

    @parallel_indices (ix, iy, iz) function _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)
        if (ix ≤ size(Vx, 1)) && (iy ≤ size(Vx, 2)) && (iz ≤ size(Vx, 3))
            Vx[ix, iy, iz] = _velocity_x(xv[ix], yc[iy], zc[iz])
        end
        if (ix ≤ size(Vy, 1)) && (iy ≤ size(Vy, 2)) && (iz ≤ size(Vy, 3))
            Vy[ix, iy, iz] = _velocity_y(xc[ix], yv[iy], zc[iz])
        end
        if (ix ≤ size(Vz, 1)) && (iy ≤ size(Vz, 2)) && (iz ≤ size(Vz, 3))
            Vz[ix, iy, iz] = _velocity_z(xc[ix], yc[iy], zv[iz])
        end

        return nothing
    end

    @parallel _velocity!(Vx, Vy, Vz, xc, yc, zc, xv, yv, zv)

    return Vx, Vy, Vz
end

function analytical_pressure(xci)
    nci = length.(xci)
    x, y, z = xci
    P = PTArray(backend)(zeros(nci[1], nci[2], nci[3]))

    @parallel_indices (ix, iy, iz) function _pressure(P, x, y, z)
        P[ix, iy, iz] = x[ix] * y[iy] * z[iz] + x[ix]^3 * y[iy]^3 * z[iz] - 5 / 32
        return nothing
    end

    @parallel _pressure(P, x, y, z)

    return P
end

function analytical_solution(xci, xvi)
    Vx, Vy, Vz = analytical_velocity(xci, xvi)
    P = analytical_pressure(xci)

    return Vx, Vy, Vz, P
end

function plot(stokes::JustRelax.StokesArrays, geometry; cmap = :vik)
    xci, xvi = geometry.xci, geometry.xvi
    vx, vy, vz, p = analytical_solution(xci, xvi)

    # numerical fields trimmed to the same staggered grids as the analytical ones
    Vx = @views stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)]
    Vy = @views stokes.V.Vy[2:(end - 1), :, 2:(end - 1)]
    Vz = @views stokes.V.Vz[2:(end - 1), 2:(end - 1), :]
    # the velocity is prescribed on every face, so the pressure is only defined up to a
    # constant; both fields are compared with their mean removed
    P = stokes.P .- mean(stokes.P)
    p = p .- mean(p)

    # slice normal to x, so every panel is a function of (y, z)
    islice = geometry.ni[1] ÷ 2

    f = Figure(; size = (2000, 1500), fontsize = 20)

    panels = (
        ("Pressure", xci[2], xci[3], P[islice, :, :], p[islice, :, :]),
        ("Vx", xci[2], xci[3], Vx[islice, :, :], vx[islice, :, :]),
        ("Vy", xvi[2], xci[3], Vy[islice, :, :], vy[islice, :, :]),
        ("Vz", xci[2], xvi[3], Vz[islice, :, :], vz[islice, :, :]),
    )

    for (col, (name, xs, ys, num, ana)) in enumerate(panels)
        # a shared color range makes the two rows directly comparable
        crange = extrema((extrema(num)..., extrema(ana)...))
        ax = Axis(f[1, 2col - 1]; title = "$name numeric", xlabel = "y", ylabel = "z")
        h = heatmap!(ax, xs, ys, num; colormap = cmap, colorrange = crange)
        Colorbar(f[1, 2col], h)

        ax = Axis(f[2, 2col - 1]; title = "$name analytical", xlabel = "y", ylabel = "z")
        h = heatmap!(ax, xs, ys, ana; colormap = cmap, colorrange = crange)
        Colorbar(f[2, 2col], h)
    end

    save("Burstedde.png", f)

    return f
end

"""
    error_norms(stokes, geometry)

Discrete L2 errors `(L2_p, L2_vx, L2_vy, L2_vz)` of a `burstedde` solution against the
analytical one. Each is `sqrt(Σ e² ΔV)`, so the norms of two resolutions are directly
comparable and their ratio measures the convergence order. The pressure is only determined
up to a constant, so both pressures enter with their mean removed.
"""
function error_norms(stokes::JustRelax.StokesArrays, geometry)
    dV = prod(geometry.di)
    vx, vy, vz, p = analytical_solution(geometry.xci, geometry.xvi)

    L2(e) = sqrt(sum(abs2, e) * dV)

    L2_vx = L2(@views stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .- vx)
    L2_vy = L2(@views stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .- vy)
    L2_vz = L2(@views stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .- vz)
    L2_p = L2((stokes.P .- mean(stokes.P)) .- (p .- mean(p)))

    return L2_p, L2_vx, L2_vy, L2_vz
end
