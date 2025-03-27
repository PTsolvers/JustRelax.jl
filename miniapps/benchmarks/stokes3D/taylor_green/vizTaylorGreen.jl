function analytical_velocity(xci, xvi)
    nvi = length.(xvi)
    nci = length.(xci)
    xc, yc, zc = xci
    xv, yv, zv = xvi
    Vx = @allocate(nvi[1], nci[2], nci[3])
    Vy = @allocate(nci[1], nvi[2], nci[3])
    Vz = @allocate(nci[1], nci[2], nvi[3])

    _velocity_x(x, y, z) = -2cos(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)
    _velocity_y(x, y, z) = sin(2 * π * x) * cos(2 * π * y) * sin(2 * π * z)
    _velocity_z(x, y, z) = sin(2 * π * x) * sin(2 * π * y) * cos(2 * π * z)

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
    P = @allocate nci...

    _pressure(x, y, z) = -6 * π * sin(2 * π * x) * sin(2 * π * y) * sin(2 * π * z)

    @parallel_indices (ix, iy, iz) function pressure(P, x, y, z)
        P[ix, iy, iz] = _pressure(x[ix], y[iy], z[iz])
        return nothing
    end

    @parallel pressure(P, x, y, z)

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

    islice = geometry.ni[1] ÷ 2

    f = Figure(; size = (2600, 900), fontsize = 20)

    # Pressure
    ax = Axis(f[1, 1]; title = "Pressure numeric")
    h = heatmap!(ax, xci[1], xci[2], stokes.P[islice, :, :]; colormap = cmap)
    Colorbar(f[1, 2], h)

    ax = Axis(f[2, 1]; title = "Pressure analytical")
    h = heatmap!(ax, xci[1], xci[2], p[islice, :, :]; colormap = :vik)
    Colorbar(f[2, 2], h)

    # Vx
    ax = Axis(f[1, 3]; title = "Vx numeric")
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vx[islice, :, :]; colormap = cmap)
    Colorbar(f[1, 4], h)

    ax = Axis(f[2, 3]; title = "Vx analytical")
    h = heatmap!(ax, xvi[1], xci[2], vx[islice, :, :]; colormap = cmap)
    Colorbar(f[2, 4], h)

    # Vy
    ax = Axis(f[1, 5]; title = "Vy numeric")
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vy[islice, :, :]; colormap = cmap)
    Colorbar(f[1, 6], h)

    ax = Axis(f[2, 5]; title = "Vy analytical")
    h = heatmap!(ax, xvi[1], xci[2], vy[islice, :, :]; colormap = cmap)
    Colorbar(f[2, 6], h)

    save("TaylorGreen.png", f)

    return f
end

function error(stokes, geometry)
    gridsize = foldl(*, geometry.di)
    vx, vy, vz, p = analytical_solution(geometry.xci, geometry.xvi)

    order = 2
    L2_vx = norm(stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)] .- vx, order) * gridsize
    L2_vy = norm(stokes.V.Vy[2:(end - 1), :, 2:(end - 1)] .- vy, order) * gridsize
    L2_vz = norm(stokes.V.Vz[2:(end - 1), 2:(end - 1), :] .- vz, order) * gridsize
    L2_p = norm(stokes.P .- (p), order) * gridsize

    return L2_p, L2_vx, L2_vy, L2_vz
end
