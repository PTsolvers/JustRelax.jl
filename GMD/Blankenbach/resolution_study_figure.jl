using JLD2, CairoMakie, MathTeXEngine

CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# directory populated by Benchmark2D_DYREL.jl (`final_$(n)x$(n).jld2`)
data_dir = "Blankenbach_DYREL_data"
resolutions = (32, 64, 128, 256)
nmax = 128 #maximum(resolutions)

let
    fig = Figure(size = (1600, 700), fontsize = 20)
    colors = cgrad(:roma, length(resolutions); categorical = true)

    # --- left column: two temperature snapshots of the highest-resolution model ---
    snapshot_files = filter(
        f -> occursin("_$(nmax)x$(nmax).jld2", f) && startswith(f, "snapshot_"),
        readdir(data_dir)
    )
    snapshot_times = [parse(Int, match(r"snapshot_(\d+)Myr_", f).captures[1]) for f in snapshot_files]
    snapshot_files = snapshot_files[sortperm(snapshot_times)]

    local h_T
    for (j, f) in enumerate(snapshot_files)
        d = load(joinpath(data_dir, f))
        t_snap = round(Int, d["time"] / (1.0e6 * 365.25 * 24 * 60 * 60))
        xci = d["xci"]
        T = d["thermal"].T[2:(end - 1), 2:(end - 1)]
        ax_T = Axis(
            fig[j, 1], aspect = DataAspect(), title = L"$$t = %$(t_snap)\ \mathrm{Myr}",
            xlabel = L"$$x", ylabel = L"$$y",
            titlesize = 24, xlabelsize = 18, ylabelsize = 18,
            xticklabelsize = 16, yticklabelsize = 16,
            xticks = [0, 1.0e6], yticks = [0, -1.0e6]

        )
        h_T = heatmap!(ax_T, xci[1], xci[2], T, colormap = :lajolla, colorrange = (273, 1273))
        contour!(ax_T, xci[1], xci[2], T, levels = 273:100:1273, color = :black, linewidth = 0.5)

        panel_label = string("(", Char('a' + (j - 1)), ")")
        inset_ax = Axis(fig[j, 1], width = Relative(0.37), height = Relative(0.12), halign = :left, valign = :top) #, backgroundcolor = :gray90)
        hidedecorations!(inset_ax); hidespines!(inset_ax)
        text!(inset_ax, 0.2, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :white)

    end
    Colorbar(fig[1:2, 2], h_T; label = L"$$T \, [K]", labelsize = 20, ticklabelsize = 16)
    colsize!(fig.layout, 1, Aspect(1, 1.0)) # xci spans a square domain; match column width to row height
    colsize!(fig.layout, 2, Fixed(30))
    colgap!(fig.layout, 1, 20)

    # --- right columns: V_rms and Nu_top vs time, all resolutions ---
    ax_rms = Axis(
        fig[1, 3], xlabel = L"$$Time", ylabel = L"$$V_{RMS}",
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 14, yticklabelsize = 14,
    )
    xlims!(ax_rms, 0.0, 4550)
    ax_nu = Axis(
        fig[2, 3], xlabel = L"$$Time", ylabel = L"$$Nu_{top}",
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 14, yticklabelsize = 14,
    )
    xlims!(ax_nu, 0.0, 4550)
    for (k, n) in enumerate(resolutions)
        dk = load(joinpath(data_dir, "final_$(n)x$(n).jld2"))
        t_myr = dk["trms"] ./ (1.0e6 * (365.25 * 24 * 60 * 60))
        keep = t_myr .<= 4500  # clip all runs to the shortest run's window
        lines!(ax_rms, t_myr[keep], dk["Urms"][keep], color = colors[k], linewidth = 2, label = L"%$(n)\times%$(n)")
        lines!(ax_nu, t_myr[keep], dk["Nu_top"][keep], color = colors[k], linewidth = 2, label = L"%$(n)\times%$(n)")

        panel_label = string("(", Char('b' + (2 - 1)), ")")
        inset_ax = Axis(fig[1, 3], width = Relative(0.17), height = Relative(0.15), halign = :left, valign = :top) #, backgroundcolor = :gray90)
        hidedecorations!(inset_ax); hidespines!(inset_ax)
        text!(inset_ax, 0.15, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :black)

        panel_label = string("(", Char('c' + (2 - 1)), ")")
        inset_ax = Axis(fig[2, 3], width = Relative(0.17), height = Relative(0.15), halign = :left, valign = :top) #, backgroundcolor = :gray90)
        hidedecorations!(inset_ax); hidespines!(inset_ax)
        text!(inset_ax, 0.15, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :black)

    end
    axislegend(ax_rms, L"$$Resolution", position = :rb, framevisible = true, labelsize = 14, titlesize = 16, orientation = :horizontal)
    # axislegend(ax_nu, L"$$Resolution", position = :rb, framevisible = true, labelsize = 14, titlesize = 16, orientation = :horizontal)

    # Label(fig[0, :], L"$$Blankenbach: Resolution Study", fontsize = 22, font = :bold)

    display(fig)
    out = joinpath(@__DIR__, "Blankenbach_resolution_study")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end
