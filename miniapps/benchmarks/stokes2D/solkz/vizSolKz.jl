using ParallelStencil.FiniteDifferences2D
using ExactFieldSolutions

# reference solution: Stokes2D_SolKz_Zhong1996 (Zhong, 1996), matching the density/viscosity
# fields used in solKz_density/solKz_viscosity (SolKz.jl): ρ = -sin(km*y)*cos(3π*x), η = exp(2B*y).
# Δη maps to B via η(y=1) = exp(2B) = Δη.
solkz_params(Δη, km) = (Δη = Δη, B = log(Δη) / 2, km = km, n = 3, σ = 1.0)

@parallel_indices (i, j) function _solkz_p!(ps, xci, yci, params)
    ps[i, j] = Stokes2D_SolKz_Zhong1996((xci[i], yci[j]); params).p
    return nothing
end

@parallel_indices (i, j) function _solkz_vx!(vxs, xvi, yci, params)
    vxs[i, j] = Stokes2D_SolKz_Zhong1996((xvi[i], yci[j]); params).V.x
    return nothing
end

@parallel_indices (i, j) function _solkz_vy!(vys, xci, yvi, params)
    vys[i, j] = Stokes2D_SolKz_Zhong1996((xci[i], yvi[j]); params).V.y
    return nothing
end

function solkz_solution(geometry; Δη = 1.0e6, km = 2)
    params = solkz_params(Δη, km)
    xci, yci = geometry.xci
    xvi, yvi = geometry.xvi

    ps = @zeros(length(xci), length(yci))
    @parallel (@idx size(ps)) _solkz_p!(ps, xci, yci, params)

    vxs = @zeros(length(xvi), length(yci))
    @parallel (@idx size(vxs)) _solkz_vx!(vxs, xvi, yci, params)

    vys = @zeros(length(xci), length(yvi))
    @parallel (@idx size(vys)) _solkz_vy!(vys, xci, yvi, params)

    return (vx = vxs, vy = vys, p = ps)
end

function Li_error(geometry, stokes; order = 2, Δη = 1.0e6, km = 2)
    solk = solkz_solution(geometry; Δη = Δη, km = km)
    gridsize = reduce(*, geometry.di)

    Li(A, B; order = 2) = norm(A .- B, order)

    L2_vx = Li(stokes.V.Vx[:, 2:(end - 1)], solk.vx; order = order) * gridsize
    L2_vy = Li(stokes.V.Vy[2:(end - 1), :], solk.vy; order = order) * gridsize
    L2_p = Li(stokes.P, solk.p; order = order) * gridsize

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
        ax1, geometry.xci[1], geometry.xvi[2], stokes.V.Vy[2:(end - 1), :]; colormap = cmap
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
        log10.(err1(Array(stokes.P), Array(solk.p)));
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
    h1 = heatmap!(ax1, geometry.xvi[1], geometry.xci[2], Array(solk.vx); colormap = cmap)
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
        log10.(err1(Array(stokes.V.Vx[:, 2:(end - 1)]), Array(solk.vx)));
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
    h1 = heatmap!(ax1, geometry.xci[1], geometry.xvi[2], Array(solk.vy); colormap = cmap)
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
        log10.(err1(Array(stokes.V.Vy[2:(end - 1), :]), Array(solk.vy)));
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
