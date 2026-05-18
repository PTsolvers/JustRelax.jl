using JLD2, CairoMakie, MathTeXEngine
using ColorSchemes

data = jldopen("data_strain_rate.jld2");
x_APT = data["x_APT"]
y_APT = data["y_APT"]
εII_APT = data["εII_APT"]
εII_DYREL = data["εII_DYREL"]

cmap = :lipari
cmap = :berlin

f = Figure(size=(800, 800))
ax = Axis(f[1, 1], 
    aspect = 3,
    xlabel= L"$$ x \, [km]", 
    ylabel= L"$$ y \, [km]", 
    title = L"$$\text{APT}$$",
)
h = heatmap!(ax, x_APT./1e3, y_APT./1e3, log10.(εII_APT), colormap=cmap)
Colorbar(f[1, 2], h, label = L"$$\log_{10}\left(\dot\varepsilon_{II}\right)$$", height=200)
f

ax = Axis(f[2, 1], 
    aspect = 3,
    xlabel= L"$$ x \, [km]", 
    ylabel= L"$$ y \, [km]", 
    title = L"$$\text{DYREL}$$",
)
h = heatmap!(ax, x_APT./1e3, y_APT./1e3, log10.(εII_DYREL), colormap=cmap)
Colorbar(f[2, 2], h, label = L"$$\log_{10}\left(\dot\varepsilon_{II}\right)$$", height=200)
f

save("APT_vs_DYREL_srtrainII.png", f)