using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny)
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    x      = range(-2000, 2000, nx);
    z      = range(-660,    20, nz);
    Grid2D = CartData(xyz_grid(x,0,z))
    Phases = zeros(Int64, nx, 1, nz);
    Temp   = fill(1280.0, nx, 1, nz);
    air_thickness = 20.0
    lith   = LithosphericPhases(Layers=[air_thickness+20 80], Phases=[1 2 0])
  
    # Add left oceanic plate
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(-2000, 0), 
        zlim    =(-660.0, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = lith, 
        T       = SpreadingRateTemp(
            Tsurface    = 20,
            Tmantle     = 1280.0,
            MORside     = "left",
            SpreadingVel= 0.5,
            AgeRidge    = 0.01;
            maxAge      = 80.0
        ) 
    )

    # Add right oceanic plate
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(1500, 2000), 
        zlim    =(-660.0, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = lith, 
        T       = SpreadingRateTemp(
            Tsurface    = 20,
            Tmantle     = 1280.0,
            MORside     = "right",
            SpreadingVel= 0.5,
            AgeRidge    = 0.01;
            maxAge      = 80.0
        ) 
    )

    # Add overriding plate margin
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(0, 400), 
        zlim    =(-660.0, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[air_thickness+25 90], Phases=[3 4 0] ), 
        T       = HalfspaceCoolingTemp(
            Tsurface = 20,
            Tmantle  = 1280.0,
            Age      = 80.0
        ) 
    )

    # Add overriding plate craton
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(400, 1500), 
        zlim    =(-660.0, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[air_thickness+35 100], Phases=[3 4 0] ), 
        T       = HalfspaceCoolingTemp(
            Tsurface = 20,
            Tmantle  = 1280.0,
            Age      = 120.0
        ) 
    )
    # Add slab
    add_box!(
        Phases, 
        Temp, 
        Grid2D; 
        xlim    =(0, 300), 
        zlim    =(-660.0, 0.0), 
        Origin  = nothing, StrikeAngle=0, DipAngle=30,
        phase   = LithosphericPhases(Layers=[air_thickness+30 80], Phases=[1 2 0], Tlab=1250 ), 
        T       = HalfspaceCoolingTemp(
            Tsurface = 20,
            Tmantle  = 1280.0,
            Age      = 120.0
        ) 
    )

    Adiabat      = 0.4
    @. Temp = Temp - Grid2D.z.val .* Adiabat

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
