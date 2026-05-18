using JLD2, CairoMakie, MathTeXEngine
using ColorSchemes
Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

function get_niter(x)
    i1 = findlast("_", x)[1]+1
    i2 = findfirst(".", x)[1]-1
    niter_apt = parse(Int, x[i1:i2])
end

function sortfiles(pth) 
    x = filter(x->contains(x, "iters"), pth)
    x[sortperm(get_niter.(x))]
end

ni = 32, 64, 128, 256, 512
pth = ["/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)/walltime.jld2" for n in ni]
wt_apt = [jldopen(pth)["walltime"] for pth in pth]
pth = ["/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)/walltime.jld2" for n in ni]
wt_dr = [jldopen(pth)["walltime"] for pth in pth]

fig = Figure(size=(1200, 600), fontsize=18)
ax  = Axis(fig[1, 1], 
    # aspect = DataAspect(),
    xlabel= L"$$ timestep",
    ylabel= L"$$ Cumulative walltime (s)", 
    yscale = log10,
    title=L"$$ Walltime",
)
cbarPal = :berlin10
cmap = cgrad(colorschemes[cbarPal], 5, categorical = true)
lw  = 1.5
l1  = lines!(cumsum(wt_apt[1]), label = L"$96  \times 32$ - APT"    , linewidth = lw, linestyle = :solid, color=cmap[1])
l2  = lines!(cumsum(wt_apt[2]), label = L"$192 \times 64$ - APT"    , linewidth = lw, linestyle = :solid, color=cmap[2])
l3  = lines!(cumsum(wt_apt[3]), label = L"$384 \times 128$ - APT"   , linewidth = lw, linestyle = :solid, color=cmap[3])
l4  = lines!(cumsum(wt_apt[4]), label = L"$768 \times 256$ - APT"   , linewidth = lw, linestyle = :solid, color=cmap[4])
l5  = lines!(cumsum(wt_apt[5]), label = L"$1536 \times 512$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[5])
l6  = lines!(cumsum(wt_dr[1]) , label = L"$96  \times 32$ - PH/DR"  , linewidth = lw, linestyle = :dash , color=cmap[1])
l7  = lines!(cumsum(wt_dr[2]) , label = L"$192 \times 64$ - PH/DR"  , linewidth = lw, linestyle = :dash , color=cmap[2])
l8  = lines!(cumsum(wt_dr[3]) , label = L"$384 \times 128$ - PH/DR" , linewidth = lw, linestyle = :dash , color=cmap[3])
l9  = lines!(cumsum(wt_dr[4]) , label = L"$768 \times 256$ - PH/DR" , linewidth = lw, linestyle = :dash , color=cmap[4])
l10 = lines!(cumsum(wt_dr[5]) , label = L"$1536 \times 512$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[5])
# axislegend(ax; position = :rb)
xlims!(ax, 0, 100)
# ylims!(ax, 0, 2e3)
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

pth     = [readdir("/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)"; join=true) for n in ni]
fls_apt = sortfiles.(pth)
pth     = [readdir("/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)"; join=true) for n in ni]
fls_dr  = sortfiles.(pth)

its_apt = [
    [jldopen(x)["out"].iter for x in x] for x in fls_apt
]
its_dr = [
    [jldopen(x)["out"].err_evo_it[end] for x in x] for x in fls_dr
]

# fig = Figure(size=(600, 600))
ax3 = Axis(fig[1, 2], 
    title=L"$$ Iteration count",
    # aspect = DataAspect(),
    xlabel= L"$$ timestep", 
    ylabel= L"$$ iterations/nx", 
    # yscale = log10,
)
# lw = 2
# lines!(ax3, log10.(its_apt[1]./96 ), label = L"$96  \times 32$  - APT"  , linewidth = lw, linestyle = :solid, color=:black)
# lines!(ax3, log10.(its_apt[2]./192), label = L"$192 \times 64$  - APT"  , linewidth = lw, linestyle = :solid, color=:red)
# lines!(ax3, log10.(its_apt[3]./384), label = L"$384 \times 128$ - APT"  , linewidth = lw, linestyle = :solid, color=:lightblue)
# lines!(ax3, log10.(its_apt[4]./768), label = L"$768 \times 256$ - APT"  , linewidth = lw, linestyle = :solid, color=:lightgreen)
# lines!(ax3, log10.(its_dr[1] ./96 ), label = L"$96  \times 32$  - PH/DR", linewidth = lw, linestyle = :dash , color=:black)
# lines!(ax3, log10.(its_dr[2] ./192), label = L"$192 \times 64$  - PH/DR", linewidth = lw, linestyle = :dash , color=:red)
# lines!(ax3, log10.(its_dr[3] ./384), label = L"$384 \times 128$ - PH/DR", linewidth = lw, linestyle = :dash , color=:lightblue)
# lines!(ax3, log10.(its_dr[4] ./768), label = L"$768 \times 256$ - PH/DR", linewidth = lw, linestyle = :dash , color=:lightgreen)
lines!(ax3, its_apt[1] ./ (ni[1] .* 3), label = L"$96  \times 32$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[1])
lines!(ax3, its_apt[2] ./ (ni[2] .* 3), label = L"$192 \times 64$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[2])
lines!(ax3, its_apt[3] ./ (ni[3] .* 3), label = L"$384 \times 128$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[3])
lines!(ax3, its_apt[4] ./ (ni[4] .* 3), label = L"$768 \times 256$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[4])
lines!(ax3, its_apt[5] ./ (ni[5] .* 3), label = L"$1536 \times 512$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[5])
lines!(ax3, its_dr[1]  ./ (ni[1] .* 3), label = L"$96  \times 32$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[1])
lines!(ax3, its_dr[2]  ./ (ni[2] .* 3), label = L"$192 \times 64$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[2])
lines!(ax3, its_dr[3]  ./ (ni[3] .* 3), label = L"$384 \times 128$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[3])
lines!(ax3, its_dr[4]  ./ (ni[4] .* 3), label = L"$768 \times 256$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[4])
lines!(ax3, its_dr[5]  ./ (ni[5] .* 3), label = L"$1536 \times 512$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[5])

# axislegend(ax; position = :lt)
xlims!(ax3, 0, 100)
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


pth     = [readdir("/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)"; join=true) for n in ni]
fls_apt = sortfiles.(pth)
pth     = [readdir("/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)"; join=true) for n in ni]
fls_dr  = sortfiles.(pth)

its_apt = [
    [jldopen(x)["out"].err_evo1[end] for x in x] for x in fls_apt
]
its_dr = [
    [jldopen(x)["out"].err_evo_V[end] for x in x] for x in fls_dr
]

fig = Figure(size=(600, 600))
ax3 = Axis(fig[1, 1], 
    # title=L"$$Iteration count",
    # aspect = DataAspect(),
    xlabel= L"$$timestep", 
    ylabel= L"$$\log_{10}(err)", 
    # yscale = log10,
)

lines!(ax3, log10.(its_apt[1]), label = L"$96  \times 32$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[1])
lines!(ax3, log10.(its_apt[2]), label = L"$192 \times 64$  - APT"  , linewidth = lw, linestyle = :solid, color=cmap[2])
lines!(ax3, log10.(its_apt[3]), label = L"$384 \times 128$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[3])
lines!(ax3, log10.(its_apt[4]), label = L"$768 \times 256$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[4])
lines!(ax3, log10.(its_apt[5]), label = L"$1536 \times 512$ - APT"  , linewidth = lw, linestyle = :solid, color=cmap[5])
lines!(ax3, log10.(its_dr[1]) , label = L"$96  \times 32$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[1])
lines!(ax3, log10.(its_dr[2]) , label = L"$192 \times 64$  - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[2])
lines!(ax3, log10.(its_dr[3]) , label = L"$384 \times 128$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[3])
lines!(ax3, log10.(its_dr[4]) , label = L"$768 \times 256$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[4])
lines!(ax3, log10.(its_dr[5]) , label = L"$1536 \times 512$ - PH/DR", linewidth = lw, linestyle = :dash , color=cmap[5])
fig