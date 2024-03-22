using GeophysicalModelGenerator

function GMG_only(nx, ny, nz)
    # nx,ny,nz = 99, 33, 66
    nx, ny, nz = (nx,ny,nz) .+ 1
    x = range(-3960, 500, nx);
    y = range(0, 2640, ny);
    z = range(-660,0,    nz);
    Grid = CartData(XYZGrid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(3, nx, ny, nz);
        
    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(1350.0, nx, ny, nz);

    # #### Simple free subduction setup

    # Much of the options are explained in the 2D tutorial, which can directly be transferred to 3D.
    # Therefore, we will start with a simple subduction setup, which consists of a horizontal part that has a mid-oceanic ridge on one explained

    # We use a lithospheric structure. Note that if the lowermost layer has the same phase as the mantle, you can define `Tlab` as the lithosphere-asthenosphere boundary which will automatically adjust the phase depending on temperature
    lith = LithosphericPhases(Layers=[15 45 10], Phases=[1 2 3], Tlab=1250)
    AddBox!(Phases, Temp, Grid; xlim=(-3000,-1000), ylim=(0, 1000.0), zlim=(-60.0, 0.0), phase = lith,
            Origin=(-0,0,0),
            T=SpreadingRateTemp(SpreadingVel=3, MORside="left"), StrikeAngle=0);

    # And an an inclined part:
    AddBox!(Phases, Temp, Grid; xlim=(0,300).-1000, ylim=(0, 1000.0), zlim=(-60.0, 0.0), phase = lith, 
            # Origin=(-1000,0,0),
            T=McKenzie_subducting_slab(Tsurface=0,v_cm_yr=3), DipAngle=15, StrikeAngle=0);
    # Add them to the `CartData` dataset:
    Grid = addField(Grid,(;Phases, Temp))

    # Which looks like
    Write_Paraview(Grid,"Initial_Setup_Subduction");

    li = abs(last(x)-first(x)), abs(last(y)-first(y)), abs(last(z)-first(z))
    origin = (x[1], y[1], z[1]) .* 1e3

    return li, origin, Phases, Temp
end
