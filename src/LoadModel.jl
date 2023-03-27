# Reads the MAT file with data provided by Fabrizio (resulting from his inversions)
#
# Also uses the 3D phase diagram (P,T, d) to report rock & melt names
using MAT, GeophysicalModelGenerator, JLD2, Interpolations

vars = matread("./Data/results.mat")

lat = vars["lat"]
lon = vars["lon"]
z = vars["depth"][:]

lat1D = sort(unique(lat))
lon1D = sort(unique(lon))

nlat, nlon, nz = length(lat1D), length(lon1D), length(z)

d3D  = zeros(nlon, nlat, nz)
T3D  = zeros(nlon, nlat, nz)
ϕ3D  = zeros(nlon, nlat, nz)
Vs3D = zeros(nlon, nlat, nz)
Vp3D = zeros(nlon, nlat, nz)
ϕmelt3D = zeros(nlon, nlat, nz)
Lat = zeros(nlon, nlat, nz)
Lon = zeros(nlon, nlat, nz)
Z = zeros(nlon, nlat, nz)

for i=1:1499
        j = findall(lat1D .== lat[i])[1]
        k = findall(lon1D .== lon[i])[1]

        for iz=1:nz
                d3D[k,j,iz] = vars["d"][i,iz]
                T3D[k,j,iz] = vars["T"][i,iz]
                ϕ3D[k,j,iz] = vars["phi"][i,iz]
                Vs3D[k,j,iz] = vars["vs"][i,iz]
                Vp3D[k,j,iz] = vars["vp"][i,iz]
                ϕmelt3D[k,j,iz] = vars["melt"][i,iz]

                Lat[k,j,iz] = lat[i]
                Lon[k,j,iz] = lon[i]
                Z[k,j,iz] = z[iz]
        end

end


ind = findall(Lon .== 0)
T3D[ind] .= NaN;
Vs3D[ind] .= NaN;
Vp3D[ind] .= NaN;
d3D[ind] .= NaN;
ϕmelt3D[ind] .= NaN;
ϕ3D[ind] .= NaN;

T3D = reverse(T3D, dims=3)
Vs3D = reverse(Vs3D, dims=3)
Vp3D = reverse(Vp3D, dims=3)
d3D = reverse(d3D, dims=3)
ϕmelt3D = reverse(ϕmelt3D, dims=3)
ϕ3D = reverse(ϕ3D, dims=3)
Z3D = reverse(Z, dims=3)


P3D = 2900*10*Z3D*1000 # P in [Pa] 



# # Add TAS data
# vars_PD  = matread("Seismo/MAGEMin_PhaseDiagram_HiRes4_v7.mat")    # phase diagram data
# #vars_PD2 = matread("MAGEMin_PhaseDiagram_HiRes2.mat")    # phase diagram data

# """
#  Interpolates a 3D data set using phase diagram data
# """
# function interpolate_data(dataset::String, vars_PD, P3D, T3D, d3D)
#         x=(vars_PD["P"], vars_PD["T"], vars_PD["d"]);
#         intp_obj= Interpolations.linear_interpolation(x,vars_PD[dataset],  extrapolation_bc=Line());
#         data_3D = zeros(size(T3D))
#         for i in eachindex(data_3D)
#                 if !isnan(P3D[i]) & !isnan(T3D[i]) & !isnan(d3D[i])
#                         data_3D[i] = intp_obj.(P3D[i],T3D[i],d3D[i]) 
#                 else
#                         data_3D[i] = NaN
#                 end
#         end
#         return data_3D
# end

# PhiM_3D       = interpolate_data("Phi_melt",vars_PD, P3D, T3D, d3D) # melt content from phase diagram
# Phase_melt    = interpolate_data("TAS_melt",vars_PD, P3D, T3D, d3D) # melt content from phase diagram
# Phase_solid   = interpolate_data("TAS_solid",vars_PD, P3D, T3D, d3D) # melt content from phase diagram
# Phase_average = interpolate_data("TAS_average",vars_PD, P3D, T3D, d3D) # melt content from phase diagram
# d_melt_3D     = interpolate_data("d_melt",vars_PD, P3D, T3D, d3D) # melt content from phase diagram
# d_sol_3D      = interpolate_data("d_solid",vars_PD, P3D, T3D, d3D) # melt content from phase diagram


# ind = findall(isnan.(PhiM_3D).==false);


Lon3D, Lat3D, Z3D = LonLatDepthGrid(lon1D, lat1D,  -z[end:-1:1])
Model3D = GeoData(Lon3D, Lat3D, Z3D*km, (T = T3D, Vs = Vs3D, Vp = Vp3D, d=d3D, 
                Phi_melt = ϕmelt3D, Damage=ϕ3D,#Phi_melt_PD = PhiM_3D,
                # Phase_melt=Phase_melt, Phase_solid=Phase_solid, Phase_average=Phase_average,
                # d_melt_3=d_melt_3D, d_sol_3D=d_sol_3D
                ))

# Read x-sections
# Corner_LowerLeft  = ( 98.63878627968339, 2.92448, -100.0)
# Corner_UpperRight = (99.40263852242745, 2.28192, 0.0)
# data_profileA = Screenshot_To_GeoData("Profile_A.png",Corner_LowerLeft, Corner_UpperRight)

# Corner_LowerLeft  = ( 98.70870712401056, 2.43808, -100.0)
# Corner_UpperRight = (99.09656992084433, 2.90016, 0.0)
# data_profileB = Screenshot_To_GeoData("Profile_B.png",Corner_LowerLeft, Corner_UpperRight)

# Corner_LowerLeft  = (98.1, 1.5, 0.0)
# Corner_UpperRight = (99.6, 3.0, 0.0)
# Topo_Map = Screenshot_To_GeoData("topo_cutoff.png",Corner_LowerLeft, Corner_UpperRight)

# save
# Write_Paraview(Model3D,"Model3D")
# Write_Paraview(data_profileA,"data_profileA")
# Write_Paraview(data_profileB,"data_profileB")

# Project 2 cartesian:
proj = ProjectionPoint(; Lat=2.19, Lon=98.91)
Model3D_cart   = Convert2CartData(Model3D, proj);
# profA_cart = Convert2CartData(data_profileA, proj);
# profB_cart = Convert2CartData(data_profileB, proj);
# Topo_cart = Convert2CartData(Topo_Map, proj);


# Write_Paraview(Model3D_cart,"Model3D_cart")

# Write_Paraview(profA_cart,"profA_cart")
# Write_Paraview(profB_cart,"profB_cart")
# Write_Paraview(Topo_cart,"Topo_cart")

return Model3D_cart
