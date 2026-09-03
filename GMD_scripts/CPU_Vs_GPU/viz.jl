using JLD2, GLMakie, MathTeXEngine

GLMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
cases = (
    (
        title = L"$$2D",
        path = joinpath(@__DIR__, "Data2D"),
        cpu = (
            "SB2D_CPU_1threads_256x256.jld2",
            "SB2D_CPU_8threads_256x256.jld2",
            "SB2D_CPU_16threads_256x256.jld2",
            "SB2D_CPU_32threads_256x256.jld2",
        ),
        gpu = "SB2D_GPU_256x256.jld2",
    ),
    (
        title = L"$$3D",
        path = joinpath(@__DIR__, "Data3D"),
        cpu = (
            "SB3D_CPU_1threads_64x64x64.jld2",
            "SB3D_CPU_8threads_64x64x64.jld2",
            "SB3D_CPU_16threads_64x64x64.jld2",
            "SB3D_CPU_32threads_64x64x64.jld2",
        ),
        gpu = "SB3D_GPU_1threads_64x64x64.jld2",
    ),
)

fig = Figure(size = (1400, 700), fontsize = 20)
colors = Makie.wong_colors()

for (column, case) in enumerate(cases)
    data = load.(joinpath.(case.path, case.cpu))
    data_cuda = load(joinpath(case.path, case.gpu))
    nt = [d["nthreads"] for d in data]
    t = [d["stokes_walltimes"] for d in data]
    t_cuda = data_cuda["stokes_walltimes"]
    timesteps = eachindex(t[1])

    ax = Axis(
        fig[1, column], xlabel = L"$$Timestep", ylabel = column == 1 ? L"$$Wall-time~[s]" : "",
        title = case.title, titlesize = 24, xlabelsize = 18, ylabelsize = 18,
    )
    hidexdecorations!(ax, grid = false)
    for i in eachindex(t)
        scatterlines!(ax, timesteps, t[i], color = colors[i], linewidth = 2.5, markersize = 12, label = "$(nt[i]) threads")
    end
    scatterlines!(ax, timesteps, t_cuda, color = :black, linewidth = 2.5, linestyle = :dash, markersize = 12, label = "GH200")
    # scatterlines!(ax, timesteps, t_cuda, color = colors[length(t) + 1], linewidth = 2.5, linestyle = :dash, marker = :star4, label = "GH200")
    xlims!(ax, 1, 20)
    column == 1 && Legend(fig[3, 1:2], ax, orientation = :horizontal, framevisible = false)

    ax = Axis(fig[2, column], xlabel = L"$$Timestep", ylabel = column == 1 ? L"$$GPU~speedup" : "", xlabelsize = 18, ylabelsize = 18)
    for i in eachindex(t)
        scatterlines!(ax, timesteps, t[i] ./ t_cuda, color = colors[i], linewidth = 2.5, markersize = 12, label = "$(nt[i]) threads")
    end
    xlims!(ax, 1, 20)
end

for (column, label) in enumerate((L"$$(a)", L"$$(b)"))
    Label(fig[1, column, TopLeft()], label, fontsize = 26, font = :bold, tellwidth = false, tellheight = false)
end

# save(joinpath(@__DIR__, "CPUvsGPU.png"), fig, px_per_unit = 2)
# save(joinpath(@__DIR__, "CPUvsGPU.pdf"), fig)
fig
