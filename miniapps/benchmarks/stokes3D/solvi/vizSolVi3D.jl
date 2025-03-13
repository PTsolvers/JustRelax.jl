function plot(stokes, geometry, rc; cmap = :vik)
    xci, xvi = geometry.xci, geometry.xvi

    cx, cy = (geometry.xvi[1][end] - geometry.xvi[1][1]) / 2,
        (geometry.xvi[2][end] - geometry.xvi[2][1]) / 2
    θ = LinRange(0, 2π, 100)
    xi, yi = @.(rc * cos(θ) + cx), @.(rc * sin(θ) + cy)

    islice = geometry.ni[1] ÷ 2

    f = Figure(; size = (2600, 900), fontsize = 20)

    # Pressure
    ax = Axis(f[1, 1]; title = "Pressure numeric")
    h = heatmap!(ax, xci[1], xci[2], stokes.P[islice, :, :]; colormap = cmap)
    lines!(ax, xi, yi; linewidth = 3, color = :black)
    Colorbar(f[1, 2], h)

    # Vx
    ax = Axis(f[1, 3]; title = "Vx numeric")
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vx[islice, :, :]; colormap = cmap)
    lines!(ax, xi, yi; linewidth = 3, color = :black)
    Colorbar(f[1, 4], h)

    # Vy
    ax = Axis(f[1, 5]; title = "Vy numeric")
    h = heatmap!(ax, xvi[1], xci[2], stokes.V.Vy[islice, :, :]; colormap = cmap)
    lines!(ax, xi, yi; linewidth = 3, color = :black)
    Colorbar(f[1, 6], h)

    save("SolVi3D.png", f)

    return f
end
