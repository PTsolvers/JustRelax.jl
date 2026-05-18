# using CairoMakie
using GLMakie
using JLD2, MathTeXEngine
using ColorSchemes
Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

data_apt = jldopen("data_Schmeling_APT.jld2")
data_dr  = jldopen("data_Schmeling_DYREL.jld2")

ni = [(125, 50) .* i for i in  1:10]

wt_apt = data_apt["wt_apt"]
wt_dr = data_dr["wt_dr"]

wt_apt = [filter(!iszero, wt_apt) for wt_apt in wt_apt]
wt_dr  = [filter(!iszero, wt_dr) for wt_dr in wt_dr]

M_apt = data_apt["Myrs_apt"]
M_dr = data_dr["Myrs_dr"]

fig = Figure(size=(1200, 600), fontsize=18)
ax  = Axis(fig[1, 1], 
    # aspect = DataAspect(),
    xlabel= L"$$ timestep",
    ylabel= L"$$ Cumulative walltime (s)", 
    yscale = log10,
    title=L"$$ Walltime",
)
cbarPal = :berlin10
cmap = cgrad(colorschemes[cbarPal], 10, categorical = true)
lw  = 1.5
for i in 2:2:10
    lines!(cumsum(wt_apt[i]), label = L"APT - $(ni[i][1])x$(ni[i][2])", linewidth = lw, linestyle = :solid, color=cmap[i])
    lines!(cumsum(wt_dr[i]) , label = L"PH/DR - $(ni[i][1])x$(ni[i][2])", linewidth = lw, linestyle = :dash , color=cmap[i])
    # lines!(ax, cumsum(wt_apt[i]) ./ cumsum(wt_dr[i]), label = L"speed up - $(ni[i][1])x$(ni[i][2])", linewidth = lw, linestyle = :solid, color=cmap[i])
end
# axislegend(ax; position = :rb)
xlims!(ax, 0, 600)
# ylims!(ax, 0, 10)
fig

# # fig = Figure(size=(600, 600))
# ax2  = Axis(fig[1, 2], 
#     # aspect = DataAspect(),
#     xlabel= L"$$ timestep", 
#     ylabel= L"$$ Cumulative speed up", 
# )
# lw = 2
# lines!(cumsum(wt_apt[1]) ./ cumsum(wt_dr[1]), label = L"$96  \times 32 $", linewidth = lw, linestyle = :solid, color=:black)
# lines!(cumsum(wt_apt[2]) ./ cumsum(wt_dr[2]), label = L"$192 \times 64 $", linewidth = lw, linestyle = :solid, color=:red)
# lines!(cumsum(wt_apt[3]) ./ cumsum(wt_dr[3]), label = L"$384 \times 128$", linewidth = lw, linestyle = :solid, color=:lightblue)
# lines!(cumsum(wt_apt[4]) ./ cumsum(wt_dr[4]), label = L"$768 \times 256$", linewidth = lw, linestyle = :solid, color=:lightgreen)
# # axislegend(ax2; position = :lt)
# xlims!(ax, 0, 100)
# # ylims!(ax, 0, 2e3)
# fig

its_apt = data_apt["its_apt"]
its_dr = data_dr["its_dr"]

# fig = Figure(size=(600, 600))
ax3 = Axis(fig[1, 2], 
    title=L"$$ Iteration count",
    # aspect = DataAspect(),
    xlabel= L"$$ timestep", 
    ylabel= L"$$ iterations/nx", 
    # yscale = log10,
)

for i in 1:2:10
    lines!(ax3, its_apt[i] ./ ni[i][1], label = L"APT - $(ni[i][1])x$(ni[i][2])", linewidth = lw, linestyle = :solid, color=cmap[i])
    lines!(ax3, its_dr[i]  ./ ni[i][1] , label = L"PH/DR - $(ni[i][1])x$(ni[i][2])", linewidth = lw, linestyle = :dash , color=cmap[i])
end
# axislegend(ax; position = :lt)
# xlims!(ax3, 0, 100)
# ylims!(ax, 0, 2e3)

methods = [
    [l1, l2, l3, l4, l5],
    [l6, l7, l8, l9, l10]
]
strings = [
        [L"$96  \times 32$"  , L"$192 \times 64$"  , L"$384 \times 128$", L"$768 \times 256$", L"$1536 \times 512$"],
        [L"$96  \times 32$"  , L"$192 \times 64$"  , L"$384 \times 128$", L"$768 \times 256$", L"$1536 \times 512$"]
]
titles = [L"$$ APT", L"$$ PH/DR"]

legs = Legend(fig,
    methods,
    strings,
    titles,
    # orientation = :horizontal,
    tellwidth = true,
    tellheight = true,
    # nbanks=5,
    framevisible = false,
)
# legs[1].nbanks = 2
fig[:,3] = legs
fig

save("SB_metrics.pdf", fig)


## error plot

err_apt = data_apt["err_apt"]
err_dr = data_dr["err_dr"]

fig = Figure(size=(600, 600))
ax3 = Axis(fig[1, 1], 
    # title=L"$$Iteration count",
    # aspect = DataAspect(),
    xlabel= L"$$timestep", 
    ylabel= L"$$\log_{10}(err)", 
    # yscale = log10,
)

lines!(ax3, log10.(err_apt[1]), label = L"$96  \times 32$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[1])
lines!(ax3, log10.(err_apt[2]), label = L"$192 \times 64$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[2])
lines!(ax3, log10.(err_apt[3]), label = L"$384 \times 128$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[3])
lines!(ax3, log10.(err_apt[4]), label = L"$768 \times 256$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[4])
lines!(ax3, log10.(err_apt[5]), label = L"$1536 \times 512$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[5])
lines!(ax3, log10.(err_dr[1]) , label = L"$96  \times 32$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[1])
lines!(ax3, log10.(err_dr[2]) , label = L"$192 \times 64$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[2])
lines!(ax3, log10.(err_dr[3]) , label = L"$384 \times 128$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[3])
lines!(ax3, log10.(err_dr[4]) , label = L"$768 \times 256$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[4])
lines!(ax3, log10.(err_dr[5]) , label = L"$1536 \times 512$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[5])
fig