# Collect the per-time-step iteration counts of several `run_comparison.jl` runs into one
# figure. Run each time step first, then this:
#
#   for dt in 10 25 50; do
#       DYREL_DT_KYR=$dt.0 DYREL_OUTDIR=results_dt${dt}kyr julia --project=. run_comparison.jl
#   done
#   julia --project=. plot_dt_comparison.jl [dt_kyr...]
#
# Each time step contributes `results_dt<dt>kyr/iterations.csv`; a missing one is an error
# rather than a silently omitted panel.

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, MathTeXEngine, DelimitedFiles, Statistics

CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

const DT_KYR = isempty(ARGS) ? [10, 25, 50] : parse.(Int, ARGS)

struct Run
    dt_kyr::Int
    step::Vector{Int}
    standard::Vector{Int}
    variational::Vector{Int}
end

function load_run(dt_kyr)
    path = joinpath(@__DIR__, "results_dt$(dt_kyr)kyr", "iterations.csv")
    isfile(path) || error("no results for dt = $dt_kyr kyr: $path does not exist")
    data = readdlm(path, ',', skipstart = 1)
    return Run(dt_kyr, Int.(data[:, 1]), Int.(data[:, 3]), Int.(data[:, 7]))
end

function panel_label!(ax, index)
    text!(
        ax, 0.02, 0.98, text = string("(", Char('a' + index), ")"), space = :relative,
        align = (:left, :top), fontsize = 25, color = :black
    )
    return nothing
end

runs = load_run.(DT_KYR)

let
    ncol = length(runs)
    fig = Figure(size = (1400, 1000), fontsize = 20)
    # Ends of the same colormap the resolution study uses, stepped inwards so both lines stay
    # inside the readable lightness band against a white page.
    colors = (cgrad(:roma)[0.15], cgrad(:roma)[0.85])

    iter_all = mapreduce(r -> vcat(r.standard, r.variational), vcat, runs)
    ylims_iter = (10.0^floor(log10(minimum(iter_all))), 10.0^ceil(log10(maximum(iter_all))))
    cum_max = maximum(r -> max(sum(r.standard), sum(r.variational)), runs) / 1.0e5

    log_ticks = ([1.0e3, 1.0e4, 1.0e5], [L"10^3", L"10^4", L"10^5"])

    # --- top row: iterations per time step, one panel per time step size ---
    local ax_first
    for (j, r) in enumerate(runs)
        ax = Axis(
            fig[1, j], title = L"$$dt = %$(r.dt_kyr) kyr",
            xlabel = L"$$Time step", ylabel = j == 1 ? L"$$Iterations per step" : "",
            titlesize = 24, xlabelsize = 18, ylabelsize = 18,
            xticklabelsize = 16, yticklabelsize = 16,
            yscale = log10, yticks = log_ticks,
            yminorticks = IntervalsBetween(9), yminorgridvisible = true,
        )
        ylims!(ax, ylims_iter)
        j == 1 && (ax_first = ax)

        lines!(ax, r.step, r.standard, color = colors[1], linewidth = 2, label = L"$$Sticky air")
        lines!(ax, r.step, r.variational, color = colors[2], linewidth = 2, label = L"$$Variational Stokes")

        panel_label!(ax, j - 1)
    end

    axislegend(ax_first, position = :rt, framevisible = true, labelsize = 14)

    # --- middle row: cumulative cost, shared scale so the panels compare directly ---
    for (j, r) in enumerate(runs)
        ax = Axis(
            fig[2, j], xlabel = L"$$Time step",
            ylabel = j == 1 ? L"$$Cumulative iterations $[\times 10^5]$" : "",
            xlabelsize = 18, ylabelsize = 18, xticklabelsize = 16, yticklabelsize = 16,
        )
        ylims!(ax, (0, 1.05cum_max))
        lines!(ax, r.step, cumsum(r.standard) ./ 1.0e5, color = colors[1], linewidth = 2)
        lines!(ax, r.step, cumsum(r.variational) ./ 1.0e5, color = colors[2], linewidth = 2)

        panel_label!(ax, ncol + j - 1)
    end

    # --- bottom row: cost scaling with the time step, spanning the full width ---
    ax_mean = Axis(
        fig[3, 1:ncol], xlabel = L"$$Time step size [kyr]", ylabel = L"$$Mean iterations per step",
        xlabelsize = 24, ylabelsize = 24, xticklabelsize = 14, yticklabelsize = 14,
        xscale = log10, yscale = log10,
        xticks = (Float64.(DT_KYR), [L"%$(dt)" for dt in DT_KYR]),
        yticks = (
            [5.0e3, 1.0e4, 2.0e4, 3.0e4],
            [L"5\times10^3", L"10^4", L"2\times10^4", L"3\times10^4"],
        ),
        yminorticks = IntervalsBetween(9), yminorgridvisible = true,
    )
    dts = Float64.(getproperty.(runs, :dt_kyr))
    # Log-log, so that the slope of each line is the exponent of its cost scaling with dt.
    xlims!(ax_mean, (0.85minimum(dts), 1.35maximum(dts)))
    # Pinned so every decade tick lands inside the axis and the log spacing is visible.
    ylims!(ax_mean, (4.0e3, 3.2e4))

    for (y, color, label) in (
            ([mean(r.standard) for r in runs], colors[1], L"$$Sticky air"),
            ([mean(r.variational) for r in runs], colors[2], L"$$Variational Stokes"),
        )
        lines!(ax_mean, dts, y, color = color, linewidth = 2, label = label)
        scatter!(ax_mean, dts, y, color = color, markersize = 12)
    end
    axislegend(ax_mean, position = :rb, framevisible = true, labelsize = 18)
    panel_label!(ax_mean, 2ncol)

    # Only open a viewer from a session that can use one; `julia plot_dt_comparison.jl` blocks
    # on the external viewer otherwise.
    isinteractive() && display(fig)
    out = joinpath(@__DIR__, "dt_comparison")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end
