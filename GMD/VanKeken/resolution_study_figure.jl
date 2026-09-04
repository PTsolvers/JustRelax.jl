using JLD2, CairoMakie, MathTeXEngine

CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# directory populated by VanKeken_DYREL.jl (`final_$(n)x$(n).jld2`, `snapshot_*_$(n)x$(n).jld2`)
data_dir = "VanKeken_DYREL_data"
resolutions = (32, 64, 128, 256, 512, 1024)
nmax = maximum(resolutions)

let
    fig = Figure(size = (1400, 700), fontsize = 20)
    colors = cgrad(:roma, length(resolutions); categorical = true)

    # --- top row: density snapshots of the highest-resolution model ---
    snapshot_files = filter(
        f -> occursin("_$(nmax)x$(nmax).jld2", f) && startswith(f, "snapshot_"),
        readdir(data_dir)
    )
    snapshot_times = [parse(Int, split(f, "_")[2]) for f in snapshot_files]
    order = sortperm(snapshot_times)
    snapshot_files = snapshot_files[order]

    local heat
    for (j, f) in enumerate(snapshot_files)
        d = load(joinpath(data_dir, f))
        t_snap = round(d["time"]; digits = 1)
        ylabel = j == 1 ? L"$$y" : ""
        ax = Axis(
            fig[1, j], aspect = DataAspect(), title = L"$$t = %$(t_snap)",
            xlabel = L"$$x", ylabel = ylabel,
            titlesize = 24, xlabelsize = 18, ylabelsize = 18,
            xticklabelsize = 14, yticklabelsize = 14,
        )
        heat = heatmap!(ax, d["xvi"][1], d["xvi"][2], d["ρg"], colormap = :lapaz)

        panel_label = string("(", Char('a' + (j - 1)), ")")
        inset_ax = Axis(fig[1, j], width = Relative(0.37), height = Relative(0.12), halign = :left, valign = :top)#, backgroundcolor = :gray90)
        hidedecorations!(inset_ax); hidespines!(inset_ax)
        text!(inset_ax, 0.10, 0.5, text = panel_label, space = :relative, align = (:center, :center), fontsize = 25, color = :black)

    end
    Colorbar(fig[1, length(snapshot_files) + 1], heat; label = L"$$Density", labelsize = 20, ticklabelsize = 16)

    # --- bottom row: V_rms vs time, all resolutions, spanning the full width ---
    ax_rms = Axis(
        fig[2, 1:length(snapshot_files)], xlabel = L"$$Time", ylabel = L"$$V_{RMS}",
        xlabelsize = 18, ylabelsize = 18, xticklabelsize = 14, yticklabelsize = 14,
    )
    for (k, n) in enumerate(resolutions)
        d = load(joinpath(data_dir, "final_$(n)x$(n).jld2"))
        lines!(ax_rms, d["trms"], d["Urms"], color = colors[k], linewidth = 2, label = L"%$(n)\times%$(n)")
    end
    axislegend(ax_rms, L"$$Resolution", position = :rt, framevisible = true, labelsize = 18, titlesize = 24)

    # Label(fig[0, :], L"$$VanKeken: Resolution Study", fontsize = 22, font = :bold)

    display(fig)
    out = joinpath(@__DIR__, "VanKeken_resolution_study")
    for ext in ("png", "pdf")
        # save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end
