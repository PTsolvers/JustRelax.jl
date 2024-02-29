using Test, GeoParams

Grid    =   CreateCartGrid(size=(96,96,96),x=(-50.,50.), y=(-50.,50.), z=(-200.,0.))
Temp    =   zeros(Float64, Grid.N...);
Phases  =   zeros(Int64,  Grid.N...);

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0), Origin=(0.0,0.0,0.0),
    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
    DipAngle=0.0, T=LithosphericTemp(nz=201))

@show sum(Temp[1,1,:])
@test sum(Temp[1,1,:]) ≈ 36131.638045729735

Temp    =   zeros(Float64, Grid.N...);
Phases  =   zeros(Int64,  Grid.N...);

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0), Origin=(0.0,0.0,0.0),
    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
    DipAngle=30.0, T=LithosphericTemp(nz=201))

@show sum(Temp[1,1,:])
@test sum(Temp[1,1,:]) ≈ 39209.69699735812

Temp    =   zeros(Float64, Grid.N...);
Phases  =   zeros(Int64,  Grid.N...);

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0),
    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
    DipAngle=30.0, T=LithosphericTemp(nz=201))

@show sum(Temp[1,1,:])
@test sum(Temp[1,1,:]) ≈ 41099.40514987406

Temp    =   zeros(Float64, Grid.N...);
Phases  =   zeros(Int64,  Grid.N...);

ρM=3.0e3            # Density [ kg/m^3 ]
CpM=1.0e3           # Specific heat capacity [ J/kg/K ]
kM=2.3              # Thermal conductivity [ W/m/K ]
HM=0.0              # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]
ρUC=2.7e3           # Density [ kg/m^3 ]
CpUC=1.0e3          # Specific heat capacity [ J/kg/K ]
kUC=3.0             # Thermal conductivity [ W/m/K ]
HUC=617.0e-12       # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]

rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase               =   1,
            Density             =   ConstantDensity(; ρ=ρUC),
            HeatCapacity        =   ConstantHeatCapacity(; Cp=CpUC),
            Conductivity        =   ConstantConductivity(; k=kUC),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HUC*ρUC),     # [H] = W/m^3
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase               =   2,
            Density             =   ConstantDensity(; ρ=ρM),
            HeatCapacity        =   ConstantHeatCapacity(; Cp=CpM),
            Conductivity        =   ConstantConductivity(; k=kM),
            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HM*ρM),       # [H] = W/m^3
        ),
    );

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0),
    phase=LithosphericPhases(Layers=[20 80], Phases = [1 2], Tlab=nothing), 
    DipAngle=30.0, T=LithosphericTemp(rheology=rheology,nz=201))

@show sum(Temp[1,1,:])
@test sum(Temp[1,1,:]) ≈ 40297.50545496938

# using flux lower boundary conditions
Temp    =   zeros(Float64, Grid.N...);
Phases  =   zeros(Int64,  Grid.N...);

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0),
    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
    DipAngle=30.0, T=LithosphericTemp(lbound="flux",nz=201))

@show sum(Temp[1,1,:])
@test sum(Temp[1,1,:]) ≈ 37182.86627823313
