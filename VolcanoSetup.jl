using GeophysicalModelGenerator

function volcano_setup(N)
    nx = ny = nz = N
    x = range(-50, 50, nx);
    y = range(-50, 50, ny);
    z = range(-50, 10, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(1350.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[15 45 100], Phases=[1 2 3])

    add_box!(Phases, Temp, Grid; 
        xlim=(-100, 100), 
        ylim=(-400, 400.0), 
        zlim=(-110.0, 0.0), 
        phase = lith, 
        T = HalfspaceCoolingTemp(Age=20)
    )
        
    add_volcano!(Phases, Temp, Grid;
        volcanic_phase  = 1,
        center     = (0, 0, 0),
        height     = 4,
        radius     = 10,
        crater     = 0.0,
        base       = 0.0,
        background = nothing,
        T = HalfspaceCoolingTemp(Age=20)
    )

    add_ellipsoid!(Phases, Temp, Grid; 
        cen    = (0, 0, -12.5), 
        axes   = (10, 10, 5),
        phase  = ConstantPhase(3),
    )

    @. Temp[Phases == 3] += 500
    @. Temp[Phases == 4]  = 20 
    @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))

    li = (abs(last(x)-first(x)),  abs(last(y)-first(y)), abs(last(z)-first(z)))
    origin = (x[1], y[1], z[1])

    ph      = Phases
    T       = Temp

    return li, origin, ph, T
end