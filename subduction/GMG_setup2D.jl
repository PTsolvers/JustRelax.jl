
#=
# Creating 2D numerical model setups

### Aim
The aim of this tutorial is to show you how to create 2D numerical model setups that can be used as initial setups for other codes.

=#


#=
### 2D Subduction setup

Lets start with creating a 2D model setup in cartesian coordinates, which uses the `CartData` data structure
=#
using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny)
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    x      = range(-1000, 1000, nx);
    z      = range(-660,0,    nz);
    Grid2D = CartData(xyz_grid(x,0,z))
    Phases = zeros(Int64, nx, 1, nz);
    Temp   = fill(1350.0, nx, 1, nz);
    lith   = LithosphericPhases(Layers=[15 20 55], Phases=[3 4 5], Tlab=1250)
    # mantle = LithosphericPhases(Phases=[1])

    # add_box!(Phases, Temp, Grid2D; xlim=(-1000, 1000), zlim=(-600.0, 0.0), phase = lith, T=HalfspaceCoolingTemp(Age=80));
    # Phases .= 0

    # Lets start with defining the horizontal part of the overriding plate. 
    # Note that we define this twice with different thickness to deal with the bending subduction area:
    add_box!(Phases, Temp, Grid2D; xlim=(200,1000), zlim=(-150.0, 0.0), phase = lith, T=HalfspaceCoolingTemp(Age=80));
    add_box!(Phases, Temp, Grid2D; xlim=(0,200), zlim=(-50.0, 0.0), phase = lith, T=HalfspaceCoolingTemp(Age=80));

    # The horizontal part of the oceanic plate is as before:
    v_spread_cm_yr = 3      #spreading velocity
    lith = LithosphericPhases(Layers=[15 55], Phases=[1 2], Tlab=1250)
    add_box!(Phases, Temp, Grid2D; xlim=(-1000,0.0), zlim=(-150.0, 0.0), phase = lith, T=SpreadingRateTemp(SpreadingVel=v_spread_cm_yr));

    # Yet, now we add a trench as well. The starting thermal age at the trench is that of the horizontal part of the oceanic plate:
    AgeTrench_Myrs = 1000e3/(v_spread_cm_yr/1e2)/1e6    #plate age @ trench

    # We want to add a smooth transition from a halfspace cooling 1D thermal profile to a slab that is heated by the surrounding mantle below a decoupling depth `d_decoupling`.
    T_slab = LinearWeightedTemperature( F1=HalfspaceCoolingTemp(Age=AgeTrench_Myrs), F2=McKenzie_subducting_slab(Tsurface=0,v_cm_yr=v_spread_cm_yr, Adiabat = 0.0))

    # in this case, we have a more reasonable slab thickness: 
    trench = Trench(Start=(0.0,-100.0), End=(0.0,100.0), Thickness=100.0, θ_max=30.0, Length=600, Lb=200, 
                    WeakzoneThickness=15, WeakzonePhase=6, d_decoupling=125);
    add_slab!(Phases, Temp, Grid2D, trench, phase = lith, T=T_slab);

    # Lithosphere-asthenosphere boundary:
    ind = findall(Temp .> 1250 .&& (Phases.==2 .|| Phases.==5));
    Phases[ind] .= 0;

    Grid2D = addfield(Grid2D,(;Phases, Temp))
   
    li = abs(last(x)-first(x)), abs(last(z)-first(z))
    origin = (x[1], z[1]) .* 1e3

    ph       = Phases[:,1,:] .+ 1
    ph2      = ph .== 2
    ph3      = ph .== 3
    ph4      = ph .== 4
    ph5      = ph .== 5
    ph6      = ph .== 6
    ph7      = ph .== 7
    ph[ph2] .= 1
    ph[ph3] .= 1
    ph[ph4] .= 2
    ph[ph5] .= 3
    ph[ph6] .= 1
    ph[ph7] .= 4

    return li, origin, ph, Temp[:,1,:]
end
   

li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx+1, ny+1)
f,ax,h=heatmap(phases_GMG)
Colorbar(f[1,2], h); f