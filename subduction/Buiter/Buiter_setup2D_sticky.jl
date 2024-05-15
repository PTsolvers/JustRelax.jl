using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny)
    model_depth = 660
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    Tbot   = 1474.0 
    x      = range(0, 3000, nx);
    z      = range(-model_depth, 10, nz);
    Grid2D = CartData(xyz_grid(x,0,z))
    Phases = zeros(Int64, nx, 1, nz);
    Temp   = fill(Tbot, nx, 1, nz);
    air_thickness = 20.0
    lith   = LithosphericPhases(Layers=[80], Phases=[1 0])
  
    # phases
    # 0: asthenosphere
    # 1: lithosphere 
    # 2: subduction lithosphere 
    # 3: oceanic crust
    # 4: air
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(0, 3000),
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[], Phases=[0]), 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=80, Adiabat=0.4)
    )

    # Add left oceanic plate
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(100, 3000-100),
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = lith, 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=80, Adiabat=0.4)
    )

    # Add right oceanic plate crust
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(3000-1430, 3000-200), 
        zlim    =(-model_depth, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[8 72], Phases=[2 1 0]), 
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=Tbot, Age=80, Adiabat=0.4)
    )

    # Add slab
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    = (3000-1430, 3000-1430-250), 
        zlim    =(-80, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=-30,
        phase   = LithosphericPhases(Layers=[8 80], Phases=[2 1 0]), #, Tlab=Tbot ), 
        T       = HalfspaceCoolingTemp(
            Tsurface = 20,
            Tmantle  = Tbot,
            Age      = 80,
        ) 
    )
    heatmap(x,z,Temp[:,1,:])
    # Lithosphere-asthenosphere boundary:
    # ind = findall(Temp .> 1250 .&& (Phases.==2 .|| Phases.==5));
    # Phases[ind] .= 0;

    surf = Grid2D.z.val .> 0.0 
    Temp[surf] .= 20.0
    Phases[surf] .= 3

    Grid2D = addfield(Grid2D,(;Phases, Temp))
   
    li = (abs(last(x)-first(x)), abs(last(z)-first(z))).* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph       = Phases[:,1,:] .+ 1

    return li, origin, ph, Temp[:,1,:].+273
end