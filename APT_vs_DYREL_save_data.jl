using JLD2, ReadVTK, GeoParams
# function get_niter(x)
#     i1 = findlast("_", x)[1]+1
#     i2 = findfirst(".", x)[1]-1
#     niter_apt = parse(Int, x[i1:i2])
# end

# function sortfiles(pth) 
#     x = filter(x->contains(x, "iters"), pth)
#     x[sortperm(get_niter.(x))]
# end

# ni = 32, 64, 128, 256, 512
# pth = ["/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)/walltime.jld2" for n in ni]
# wt_apt = [jldopen(pth)["walltime"] for pth in pth]
# pth = ["/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)/walltime.jld2" for n in ni]
# wt_dr = [jldopen(pth)["walltime"] for pth in pth]

# pth     = [readdir("/Users/albert/Documents/Duretz_APT_$(n*3)x$(n)"; join=true) for n in ni]
# fls_apt = sortfiles.(pth)
# pth     = [readdir("/Users/albert/Documents/Duretz_DYREL_$(n*3)x$(n)"; join=true) for n in ni]
# fls_dr  = sortfiles.(pth)

# its_apt = [
#     [jldopen(x)["out"].iter for x in x] for x in fls_apt
# ]
# its_dr = [
#     [jldopen(x)["out"].err_evo_it[end] for x in x] for x in fls_dr
# ]


# err_apt = [
#     [jldopen(x)["out"].err_evo1[end] for x in x] for x in fls_apt
# ]
# err_dr = [
#     [jldopen(x)["out"].err_evo_V[end] for x in x] for x in fls_dr
# ]

# jldsave(
#     "data_APT.jld2";
#     ni,
#     wt_apt,
#     its_apt,
#     err_apt,
# )

# jldsave(
#     "data_DYREL.jld2";
#     ni,
#     wt_dr,
#     its_dr,
#     err_dr,
# )

# using JLD2, CairoMakie, MathTeXEngine
# using ColorSchemes
# using ReadVTK
# using GeoParams

# Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# cell_average(A) = 0.25 .* (A[1:end-1, 1:end-1] .+ A[2:end, 1:end-1] .+ A[1:end-1, 2:end] .+ A[2:end, 2:end])

# function second_inv(εxx_vtk, εyy_vtk, εxy_vtk)
#     εII = zeros(n*3, n)
#     for i in 1:n*3, j in 1:n
#         εxx, εyy = Float64(εxx_vtk[i, j]), Float64(εyy_vtk[i, j])
#         εxy =  εxy_vtk[i, j],  εxy_vtk[i+1, j],  εxy_vtk[i, j+1],  εxy_vtk[i+1, j+1]
#         εII[i, j] = second_invariant_staggered(εxx, εyy, εxy) 
#     end
#     return εII
# end

## strain rate
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
jldsave(
    "data_strain_rate.jld2";
    x_APT, y_APT, εII_APT, εII_DYREL,
)