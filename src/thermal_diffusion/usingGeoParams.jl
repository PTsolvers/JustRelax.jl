using GeoParams

xmin,xmax   =   -50.0,50.0
ymin,ymax   =   xmin, xmax
zmin,zmax   =   -200.0,0.0
nel     =   96
x       =   LinRange(xmin,xmax,nel)
y       =   LinRange(ymin,ymax,nel)
z       =   LinRange(zmin,zmax,nel)
X,Y,Z   =   XYZGrid(x, y, z)
Data    =   Z
Grid =  CartData(X,Y,Z,(FakeData=Data,Data2=Data.+1.))

Phases = zeros(Int64,   size(Grid.x));
Temp   = zeros(Float64, size(Grid.x));

#ρM=3.0e3            # Density [ kg/m^3 ]
#CpM=1.0e3           # Specific heat capacity [ J/kg/K ]
#kM=2.3              # Thermal conductivity [ W/m/K ]
#HM=0.0              # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]
#ρUC=2.7e3           # Density [ kg/m^3 ]
#CpUC=1.0e3          # Specific heat capacity [ J/kg/K ]
#kUC=3.0             # Thermal conductivity [ W/m/K ]
#HUC=617.0e-12       # Radiogenic heat source per mass [H] = W/kg; [H] = [Q/rho]

#rheology = (
#        # Name              = "UpperCrust",
#        SetMaterialParams(;
#            Phase               =   1,
#            Density             =   ConstantDensity(; ρ=ρUC),
#            HeatCapacity        =   ConstantHeatCapacity(; Cp=CpUC),
#            Conductivity        =   ConstantConductivity(; k=kUC),
#            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HUC*ρUC),     # [H] = W/m^3
#        ),
#        # Name              = "LithosphericMantle",
#        SetMaterialParams(;
#            Phase               =   2,
#            Density             =   ConstantDensity(; ρ=ρM),
#            HeatCapacity        =   ConstantHeatCapacity(; Cp=CpM),
#            Conductivity        =   ConstantConductivity(; k=kM),
#            RadioactiveHeat     =   ConstantRadioactiveHeat(; H_r=HM*ρM),       # [H] = W/m^3
#        ),
#    );

AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0), Origin=(0.0,0.0,0.0),
    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
    DipAngle=0.0, T=LithosphericTemp(nz=201))

#AddBox!(Phases,Temp,Grid, xlim=(-50,50), zlim=(-100,0),
#    phase=LithosphericPhases(Layers=[20 15 65], Phases = [1 2 3], Tlab=nothing), 
#    DipAngle=30.0, T=LithosphericTemp(nz=201))

@show sum(Temp[1,1,:])

Grid = AddField(Grid, "Phases", Phases)
Grid = AddField(Grid, "Temp", Temp)
         
Write_Paraview(Grid,"/mnt/d/Users/lukas/LaMEM_ModelSetup")           