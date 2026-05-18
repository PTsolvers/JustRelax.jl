using JLD2, CairoMakie, MathTeXEngine
using ColorSchemes
using ReadVTK
using GeoParams

Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

cell_average(A) = 0.25 .* (A[1:end-1, 1:end-1] .+ A[2:end, 1:end-1] .+ A[1:end-1, 2:end] .+ A[2:end, 2:end])

function second_inv(εxx_vtk, εyy_vtk, εxy_vtk)
    εII = zeros(n*3, n)
    for i in 1:n*3, j in 1:n
        εxx, εyy = Float64(εxx_vtk[i, j]), Float64(εyy_vtk[i, j])
        εxy =  εxy_vtk[i, j],  εxy_vtk[i+1, j],  εxy_vtk[i, j+1],  εxy_vtk[i+1, j+1]
        εII[i, j] = second_invariant_staggered(εxx, εyy, εxy) 
    end
    return εII
end

n = 512
pth_APT1 = "/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)/vtk_000100_1.vti" 
pth_APT2 = "/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)/vtk_000100_2.vti" 

vtk = VTKFile(pth_APT1)
point_data = get_point_data(vtk)
vtk_v = VTKFile(pth_APT2)
point_data_v = get_point_data(vtk_v)

# ImageData: build coordinates from Origin and Spacing
# WholeExtent: 0 1536 0 512; Origin: (-50000, -30000); Spacing: (65.104, 58.594)
nx_APT, ny_APT = n*3, n
xml_img  = ReadVTK.LightXML.root(vtk.xml_file)["ImageData"][1]
origin   = parse.(Float64, split(ReadVTK.LightXML.attribute(xml_img, "Origin")))
spacing  = parse.(Float64, split(ReadVTK.LightXML.attribute(xml_img, "Spacing")))
x_APT    = range(origin[1], step=spacing[1], length=n*3)
y_APT    = range(origin[2], step=spacing[2], length=n)

# read point data fields
εxx_APT  = reshape(get_data(point_data["εxx"]),  n*3, n)
εyy_APT  = reshape(get_data(point_data["εyy"]),  n*3, n)
εxy_APT  = reshape(get_data(point_data_v["εxy"]),  n*3+1, n+1)
εII_APT = second_inv(εxx_APT, εyy_APT, εxy_APT)

pth_APT1 = "/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)/vtk_000100_1.vti" 
pth_APT2 = "/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)/vtk_000100_2.vti" 

vtk = VTKFile(pth_APT1)
point_data = get_point_data(vtk)
vtk_v = VTKFile(pth_APT2)
point_data_v = get_point_data(vtk_v)

# read point data fields
εxx_DYREL  = reshape(get_data(point_data["εxx"]),  n*3, n)
εyy_DYREL  = reshape(get_data(point_data["εyy"]),  n*3, n)
εxy_DYREL  = reshape(get_data(point_data_v["εxy"]),  n*3+1, n+1)
εII_DYREL = second_inv(εxx_DYREL, εyy_DYREL, εxy_DYREL)

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