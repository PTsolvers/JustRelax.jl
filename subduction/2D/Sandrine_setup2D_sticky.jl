using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny)
    model_depth = 1_000
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    Tbot   = 1794.0 
    x      = range(0, 3000, nx);
    z      = range(-model_depth, 10, nz);
    Grid2D = CartData(xyz_grid(x,0,z))
    Phases = zeros(Int64, nx, 1, nz);
    Temp   = fill(Tbot, nx, 1, nz);
    air_thickness = 20.0
    lith   = LithosphericPhases(Layers=[3 3 5], Phases=[1 2 3 0])
  
    # phases
    # 0: asthenosphere
    # 1: sediments (wet qz.)
    # 2: oceanic crust (basalt)
    # 3: oceanic crust (An75)
    # 4: weak zone (wet ol.)
    # 5: air

    # Add left oceanic plate
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(0, 3000), 
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = lith, 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=1, Adiabat=0.4)
    )

    # Add right oceanic plate
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(1155, 1375), 
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[3 8], Phases=[1 2 0]), 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=80, Adiabat=0.4)
    )
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(1375, 3000), 
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = lith, 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=80, Adiabat=0.4)
    )

    # Add weak zone
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(1092, 1155), 
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[3 60], Phases=[1 4 0]), 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=1, Adiabat=0.4)
    )

    # Adiabat      = 0.4
    # @. Temp = Temp - Grid2D.z.val .* Adiabat

    # Lithosphere-asthenosphere boundary:
    # ind = findall(Temp .> 1250 .&& (Phases.==2 .|| Phases.==5));
    # Phases[ind] .= 0;

    surf = Grid2D.z.val .> 0.0 
    Temp[surf] .= 20.0
    Phases[surf] .= 5

    Grid2D = addfield(Grid2D,(;Phases, Temp))
   
    li = (abs(last(x)-first(x)), abs(last(z)-first(z))).* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph       = Phases[:,1,:] .+ 1

    return li, origin, ph, Temp[:,1,:].+273
end