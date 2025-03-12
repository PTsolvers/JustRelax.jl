using ParallelStencil.FiniteDifferences2D

include("SolKz_solution.jl")

function solkz_solution(geometry)
    # element center
    xci, yci = geometry.xci
    xc = [xc for xc in xci, _ in yci]
    yc = [yc for _ in xci, yc in yci]
    # element vertices
    xvi, yvi = geometry.xvi
    xv_x = [xc for xc in xvi, _ in yci] # for vx
    yv_x = [yc for _ in xvi, yc in yci] # for vx

    xv_y = [xc for xc in xci, _ in yvi] # for vy
    yv_y = [yc for _ in xci, yc in yvi] # for vy

    # analytical solution
    ps = similar(xc) # @ centers
    vxs = similar(xv_x) # @ vertices
    vys = similar(xv_y) # @ vertices
    Threads.@threads for i in eachindex(xc)
        @inbounds _, _, ps[i] = _solkz_solution(xc[i], yc[i])
    end
    Threads.@threads for i in eachindex(xv_x)
        @inbounds vxs[i], = _solkz_solution(xv_x[i], yv_x[i])
    end
    Threads.@threads for i in eachindex(xv_y)
        @inbounds _, vys[i], = _solkz_solution(xv_y[i], yv_y[i])
    end

    return (vx = (vxs), vy = (vys), p = (ps))
end

function Li_error(geometry, stokes; order = 2)
    solk = solkz_solution(geometry)
    gridsize = reduce(*, geometry.di)

    Li(A, B; order = 2) = norm(A .- B, order)

    L2_vx = Li(stokes.V.Vx[:, 2:(end - 1)], PTArray(backend)(solk.vx); order = order) * gridsize
    L2_vy = Li(stokes.V.Vy[2:(end - 1), :], PTArray(backend)(solk.vy); order = order) * gridsize
    L2_p = Li(stokes.P, PTArray(backend)(solk.p); order = order) * gridsize

    return L2_vx, L2_vy, L2_p
end

function plot_solkz(geometry, ρ, stokes; cmap = :vik)
    f = Figure(; size = (3000, 1800), fontsize = 28)

    #Ddensity
    ax1 = Axis(f[1, 1]; aspect = 1)
    h1 = heatmap!(ax1, geometry.xci[1], geometry.xci[2], ρ; colormap = cmap)

    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[1, 2], h1; label = "density")

    # Pressure
    ax1 = Axis(f[1, 3]; aspect = 1)
    h1 = heatmap!(ax1, geometry.xci[1], geometry.xci[2], stokes.P; colormap = cmap)
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[1, 4], h1; label = "P")

    # Velocity-x
    ax1 = Axis(f[2, 1]; aspect = 1)
    h1 = heatmap!(
        ax1, geometry.xvi[1], geometry.xci[2], stokes.V.Vx[:, 2:(end - 1)]; colormap = cmap
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[2, 2], h1; label = "Vx")

    # Velocity-y
    ax1 = Axis(f[2, 3]; aspect = 1)
    h1 = heatmap!(
        ax1, geometry.xvi[2], geometry.xci[1], stokes.V.Vy[2:(end - 1), :]; colormap = cmap
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[2, 4], h1; label = "Vy")

    save("SolKz.png", f)

    return f
end

function plot_solKz_error(geometry, stokes; cmap = :vik)
    solk = solkz_solution(geometry)

    # Plot
    f = Figure(; size = (1200, 1000), fontsize = 20)

    # ROW 1: PRESSURE
    # Numerical pressure
    ax1 = Axis(f[1, 1]; aspect = 1, title = "numerical")
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        Array(stokes.P);
        colormap = cmap,
        colorrange = extrema(stokes.P),
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    # Colorbar(f[1,2], h1, label="Pressure")

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)

    # Analytical pressure
    ax1 = Axis(f[1, 2]; aspect = 1, title = "analytical")
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        Array(solk.p);
        colormap = cmap,
        colorrange = extrema(stokes.P),
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[1, 3], h1; label = "P", height = 300, width = 20, tellheight = true)

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # Pressure error
    ax1 = Axis(f[1, 4]; aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xci[2],
        log10.(err1(Array(stokes.P), solk.p));
        colormap = :batlow,
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[1, 5], h1; label = "log10 error P")

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # ROW 2: Velocity-x
    # Numerical
    ax1 = Axis(f[2, 1]; aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        Array(stokes.V.Vx[:, 2:(end - 1)]);
        colormap = cmap,
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)

    ax1 = Axis(f[2, 2]; aspect = 1)
    h1 = heatmap!(ax1, geometry.xvi[1], geometry.xci[2], solk.vx; colormap = cmap)
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[2, 3], h1; label = "Vx", width = 20, tellheight = true)

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    ax1 = Axis(f[2, 4]; aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xvi[1],
        geometry.xci[2],
        log10.(err1(Array(stokes.V.Vx[2:(end - 1), 2:(end - 1)]), solk.vx[2:(end - 1), :]));
        colormap = :batlow,
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[2, 5], h1; label = "log10 error Vx", height = 300, width = 20, tellheight = true)

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hidexdecorations!(ax1)
    hideydecorations!(ax1)

    # ROW 3: Velocity-y
    # Numerical
    ax1 = Axis(f[3, 1]; aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xvi[2],
        Array(stokes.V.Vy[2:(end - 1), :]);
        colormap = cmap,
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    ax1 = Axis(f[3, 2]; aspect = 1)
    h1 = heatmap!(ax1, geometry.xci[1], geometry.xvi[2], solk.vy; colormap = cmap)
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[3, 3], h1; label = "Vy", height = 300, width = 20, tellheight = true)

    ax1.xticks = 0:1
    ax1.yticks = 0:1
    hideydecorations!(ax1)

    ax1 = Axis(f[3, 4]; aspect = 1)
    h1 = heatmap!(
        ax1,
        geometry.xci[1],
        geometry.xvi[2],
        log10.(err1(Array(stokes.V.Vy[2:(end - 1), 2:(end - 1)]), solk.vy[:, 2:(end - 1)]));
        colormap = :batlow,
    )
    xlims!(ax1, (0, 1))
    ylims!(ax1, (0, 1))
    Colorbar(f[3, 5], h1; label = "log10 error Vy", height = 300, width = 20, tellheight = true)

    ax1.xticks = 0:1
    ax1.yticks = 0:1

    hideydecorations!(ax1)

    save("SolKz_error.png", f)

    return f
end

err2(A::AbstractArray, B::AbstractArray) = @. √(((A - B)^2))

err1(A::AbstractArray, B::AbstractArray) = @. abs(A - B)
