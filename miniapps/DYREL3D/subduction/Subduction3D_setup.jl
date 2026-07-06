using GeophysicalModelGenerator

function GMG_only(nx, ny, nz)

    x = range(-3960, 500, nx)
    y = range(0, 2640, ny)
    air_thickness = 0.0
    z = range(-660, air_thickness, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(3, nx, ny, nz)

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(1350.0, nx, ny, nz)

    #### Simple free subduction setup

    # Much of the options are explained in the 2D tutorial, which can directly be transferred to 3D.
    # Therefore, we will start with a simple subduction setup, which consists of a horizontal part that has a mid-oceanic ridge on one explained

    # We use a lithospheric structure. Note that if the lowermost layer has the same phase as the mantle, you can define `Tlab` as the lithosphere-asthenosphere boundary which will automatically adjust the phase depending on temperature
    lith = LithosphericPhases(Layers = [80], Phases = [1 2], Tlab = 1250)
    # add_box!(Phases, Temp, Grid; xlim=(-3000,-1000), ylim=(0, 1000.0), zlim=(-80.0, 0.0), phase = lith,
    #         Origin=(-0,0,0),
    #         T=SpreadingRateTemp(SpreadingVel=3, MORside="left"), StrikeAngle=0);

    # # And an an inclined part:
    # lith = LithosphericPhases(Layers=[200], Phases=[1 2], Tlab=1250)
    # add_box!(Phases, Temp, Grid; xlim=(0,300).-1000, ylim=(0, 1000.0), zlim=(-80.0, 10.0), phase = lith,
    #         # Origin=(-1000,0,0),
    #         T=SpreadingRateTemp(SpreadingVel=0, MORside="left"), DipAngle=20, StrikeAngle=0);

    add_box!(Phases, Temp, Grid; xlim = (-3000, -1000), ylim = (0, 1000), zlim = (-80, 0), phase = LithosphericPhases(Layers = [20, 60], Phases = [1, 2]))
    add_box!(Phases, Temp, Grid, xlim = (-1000, -810), ylim = (0, 1000), zlim = (-80, 0), phase = LithosphericPhases(Layers = [20, 60], Phases = [1, 2]), DipAngle = 20)

    # Add them to the `CartData` dataset:
    Grid = addfield(Grid, (; Phases, Temp))

    # Which looks like
    write_paraview(Grid, "Initial_Setup_Subduction")

    surf = Grid.z.val .> 0.0
    Phases[surf] .= 4

    li = (abs(last(x) - first(x)), abs(last(y) - first(y)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], y[1], z[1]) .* 1.0e3

    return li, origin, Phases, Temp
end
