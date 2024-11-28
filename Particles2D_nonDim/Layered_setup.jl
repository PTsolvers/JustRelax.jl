using GeophysicalModelGenerator

function setup(nx, ny)
    model_depth   = 2800
    Lx            = 40 / 4
    Lx            = 6000
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz        = nx, ny
    Tbot          = 1474.0
    x             = range(0, Lx, nx);
    air_thickness = 0e0
    z             = range(-model_depth, air_thickness, nz);
    Grid2D        = CartData(xyz_grid(x, 0, z))
    Phases        = zeros(Int64, nx, 1, nz);
    Temp          = fill(Tbot, nx, 1, nz);

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
        xlim    = (0, Lx),
        zlim    = (-model_depth, 0.0),
        # Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[20, 40, 120, 660, 2600], Phases=[1, 2, 3, 4, 5, 6]),
        T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=2000, Age=1000, Adiabat=0.4)
    )

    ## add a sphere
    add_ellipsoid!(
        Phases,
        Temp,
        Grid2D;
        cen      = (107.5, 0, -100.0),
        axes     = (10, 10, 1.5),
        DipAngle = 45,
        phase    = ConstantPhase(3),
        # T        = ConstantTemp(T=1200),
    )

    # Grid2D = addfield(Grid2D,(;Phases, Temp))
    # write_paraview(Grid2D,"Initial_Setup_Subduction_rank");
    li = (abs(last(x)-first(x)), abs(last(z)-first(z))).* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:] .+ 273
    # heatmap(x, z, ph)
    # heatmap(x, z, T)

    return li, origin, ph, T
end

function setup_vertical(nx, ny)
    Lx            = 40 
    model_depth   = 165
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz        = nx, ny
    Tbot          = 1474.0
    x             = range(0, Lx, nx);
    air_thickness = 0e0
    z             = range(-model_depth, air_thickness, nz);
    Grid2D        = CartData(xyz_grid(x, 0, z))
    Phases        = zeros(Int64, nx, 1, nz);
    Temp          = fill(Tbot, nx, 1, nz);
    Tlab          = 800
    # lith   = LithosphericPhases(Layers=[80], Phases=[1 0], Tlab=Tlab)

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
        xlim    = (0, Lx),
        zlim    = (-model_depth, 0.0),
        # Origin  = nothing, StrikeAngle=0, DipAngle=0,
        phase   = LithosphericPhases(Layers=[], Phases=[1], Tlab=Tlab),
        T       = LinearTemp(Ttop=500, Tbot=770)
    )

    # Add left oceanic plate
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim    =(Lx/2 - 2.5, Lx/2 + 2.5),
        zlim    =(-model_depth, 0),
        # Origin  = nothing, StrikeAngle=0, DipAngle=45,
        phase   = LithosphericPhases(Layers=[200], Phases=[2], Tlab=Tlab),
        # T       = HalfspaceCoolingTemp(Tsurface=20, Tmantle=770, Age=100, Adiabat=0)
    )

    ## add a sphere
    add_ellipsoid!(
        Phases,
        Temp,
        Grid2D;
        cen      = (Lx/2, 0, -140.0),
        axes     = (1.5, 3, 10),
        # DipAngle = 45,
        phase    = ConstantPhase(3),
        # T        = ConstantTemp(T=1200),
    )
    heatmap(Phases[:,1,:])
    # surf = Grid2D.z.val .> 0.0
    # Temp[surf] .= 20.0
    # Phases[surf] .= 3

    # Grid2D = addfield(Grid2D,(;Phases, Temp))
    # write_paraview(Grid2D,"Initial_Setup_Subduction_rank");
    li = (abs(last(x)-first(x)), abs(last(z)-first(z))).* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:] .+ 273

    return li, origin, ph, T
end

