using ExactFieldSolutions

# Analytical solution found in:
#     D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
#     general shear. Geophysical Journal International, 155(1):269–288, 2003.

function solvi_solution(geometry, Δη, rc, εbg)
    # Pressure
    x = geometry.xci[1]
    x = x .- (x[end] - x[1]) / 2
    y = geometry.xci[2]
    y = y .- (y[end] - y[1]) / 2
    X = [x for x in x, y in y]
    Y = [y for x in x, y in y]

    P = zeros(size(X))

    for i in eachindex(P)
        sol = Stokes2D_Schmid2003(
            [X[i], Y[i]];
            params = (mm = 1.0, mc = 1.0e-3, rc = 0.2, gr = 0.0, er = 1)
        )
        P[i] = sol.p
    end

    # Vx
    x = geometry.xvi[1]
    x = x .- (x[end] - x[1]) / 2
    y = geometry.xci[2]
    y = y .- (y[end] - y[1]) / 2
    X = [x for x in x, y in y]
    Y = [y for x in x, y in y]

    Vx = zeros(size(X))

    for i in eachindex(Vx)
        sol = Stokes2D_Schmid2003(
            [X[i], Y[i]];
            params = (mm = 1.0, mc = 1.0e-3, rc = 0.2, gr = 0.0, er = -1)
        )
        Vx[i] = sol.V[1]
    end

    # Vy
    x = geometry.xci[1]
    x = x .- (x[end] - x[1]) / 2
    y = geometry.xvi[2]
    y = y .- (y[end] - y[1]) / 2
    X = [x for x in x, y in y]
    Y = [y for x in x, y in y]

    Vy = zeros(size(X))

    for i in eachindex(Vy)
        sol = Stokes2D_Schmid2003(
            [X[i], Y[i]];
            params = (mm = 1.0, mc = 1.0e-3, rc = 0.2, gr = 0.0, er = -1)
        )
        Vy[i] = sol.V[2]
    end
    return (p = P, vx = Vx, vy = Vy)
end

function Li_error(geometry, stokes, Δη, εbg, rc, ; order = 2)

    # analytical solution
    sol = solvi_solution(geometry, Δη, rc)
    gridsize = reduce(*, geometry.di)

    Li(A, B; order = 2) = norm(A .- B, order)

    L2_vx = Li(stokes.V.Vx[:, 2:(end - 1)], PTArray(sol.vx); order = order) * gridsize
    L2_vy = Li(stokes.V.Vy[2:(end - 1), :], PTArray(sol.vy); order = order) * gridsize
    L2_p = Li(stokes.P, PTArray(sol.p); order = order) * gridsize

    return L2_vx, L2_vy, L2_p
end

function plot_solVi_error(geometry, stokes, Δη, εbg, rc)

    # analytical solution
    sol = solvi_solution(geometry, Δη, εbg, rc)

    cx, cy = (geometry.xvi[1][end] - geometry.xvi[1][1]) / 2,
        (geometry.xvi[2][end] - geometry.xvi[2][1]) / 2
    θ = LinRange(0, 2π, 100)
    ix, iy = @.(rc * cos(θ) + cx), @.(rc * sin(θ) + cy)

    f = Figure(; size = (1200, 1200), fontsize = 20)

    # Pressure plots
    ax1 = Axis(f[1, 1]; title = "P numeric", aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        stokes.P;
        colorrange = extrema(stokes.P),
        colormap = :romaO,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)

    hidexdecorations!(ax1)

    ax1 = Axis(f[1, 2]; title = "P analytical", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        sol.p';
        colorrange = extrema(stokes.P),
        colormap = :romaO,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[1, 3], h; height = 300)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    ax1 = Axis(f[1, 4]; title = "P error", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        @.(log10(sqrt((stokes.P' - sol.p)^2)));
        colormap = Reverse(:batlow),
        colorrange = (-7, 1),
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[1, 5], h)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # Velocity-x plots
    ax1 = Axis(f[2, 1]; title = "Vx numeric", aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        stokes.V.Vx;
        # colorrange=(-1, 1),
        colormap = :romaO,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)

    hidexdecorations!(ax1)

    ax1 = Axis(f[2, 2]; title = "Vx analytical", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        sol.vx;
        # colorrange=(-1, 1),
        colormap = :romaO,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[2, 3], h; height = 300)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    ax1 = Axis(f[2, 4]; title = "Vx error", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        log10.(err2(Array(stokes.V.Vx[:, 2:(end - 1)]), sol.vx));
        colormap = :batlow,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[2, 5], h)

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # Velocity-z plots
    ax1 = Axis(f[3, 1]; title = "Vy numeric", aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        stokes.V.Vy;
        # colorrange=(-1, 1),
        colormap = :romaO,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)

    ax1 = Axis(f[3, 2]; title = "Vy analytical", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        sol.vy;
        colormap = :romaO
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[3, 3], h; height = 300)

    hideydecorations!(ax1)

    ax1 = Axis(f[3, 4]; title = "Vy error", aspect = 1)
    h = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        log10.(err2(Array(stokes.V.Vy[2:(end - 1), :]), sol.vy));
        colormap = :batlow,
    )
    lines!(ax1, ix, iy; linewidth = 3, color = :black)
    Colorbar(f[3, 5], h)

    hideydecorations!(ax1)

    save("SolVi.png", f)

    return f
end

err2(A::AbstractArray, B::AbstractArray) = @. √(((A - B)^2))

err1(A::AbstractArray, B::AbstractArray) = @. abs(A - B)
