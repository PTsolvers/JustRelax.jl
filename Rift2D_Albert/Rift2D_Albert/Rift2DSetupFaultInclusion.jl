using GeophysicalModelGenerator

function flat_setup(Nx, Nz)

    Lx = Ly = 500
    Lz = 110
    h_air = 10
    nx = Nx 
    nz = Nz
    ny = 10
    x = range(-Lx / 2, Lx / 2, nx);
    y = range(-eps(), eps(), ny);
    # z = range(-18, 10, nz);
    z = range(-Lz, h_air, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, ny, nz);

    depth_layers = [22 40 100] # depth of the ith layer in km
    lith = LithosphericPhases(Layers=depth_layers, Phases=[1 2 3 4])

    add_box!(Phases, Temp, Grid; 
        xlim  =(-Lx, Lx), 
        ylim  =(-Ly, Ly), 
        zlim  =(-Lz, 0.0), 
        phase = lith, 
        T     = HalfspaceCoolingTemp(
            Age      = 70,
            Tmantle  = 1330,
            Tsurface = 0,
        )
    )
    
    for I in eachindex(Grid.z.val)
        if Grid.z.val[I...] > 0 
            Phases[I...] = 5 # air
            Temp[I...] = 0
        end
    end

    Grid = addfield(Grid,(; Phases, Temp))

    li     = (abs(last(x)-first(x)),  abs(last(z)-first(z))) .* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases
    T       = Temp

    write_paraview(Grid,"Rift2D_Setup_FaultInclusion");

    return li, origin, ph[:,1,:], T[:,1,:] .+ 273
end