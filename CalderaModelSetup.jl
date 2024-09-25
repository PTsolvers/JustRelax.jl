## Model Setup

function Toba_topo(nx,ny,nz)
    #import topo
    Topo = load("./Data/Toba_Topography_Jul24.jld2", "TopoToba_highres") # 3s resolution
    # Topo = load("./Data/Toba_Topography_Jul24.jld2", "TopoToba_lowres") # 30s resolution
    # extract 2D array with the topographic elevation
    surf = Topo.depth.val[:,:,1]
    # l = maximum(abs.(Topo.depth.val))
    proj = ProjectionPoint(; Lat=2.19, Lon=98.91);
    cartesian = convert2CartData(Topo, proj)

    # our model is in [km] -> need to convert from lat-long
    # x = LinRange(minimum(Topo.lon.val), maximum(Topo.lon.val), nx) |> collect
    # y = LinRange(minimum(Topo.lat.val), maximum(Topo.lat.val), ny) |> collect
    # our model is in [km] -> need to convert from lat-long
    x = LinRange(98.0, 99.8, nx) |> collect
    y = LinRange(1.2, 3.3, nz) |> collect
    X = (
        Topo.lon.val[:, 1, 1],
        Topo.lat.val[1, :, 1],
    )

    # interpolate from GMT's surface to the resolution nx × ny resolution of our grid
    itp = interpolate(X, surf, Gridded(Linear()))
    surf_interp = [itp(x, y) for x in x, y in y]

    # compute the x and y size of our cartesian model
    lat_dist  = extrema(X[1]) |> collect |> diff |> first |> abs
    long_dist = extrema(X[2]) |> collect |> diff |> first |> abs
    Lx        = lat_dist * 110.574
    Ly        = long_dist * 111.320*cos(lat_dist)

    return surf_interp, cartesian, Lx, Ly
end

function Toba_setup2D(nx,ny,nz; sticky_air=5)
    topo_toba, Topo_cartesian, Lx, Ly = Toba_topo(nx,ny,nz)

    Grid3D = create_CartGrid(;
        size=(nx, ny, nz),
        x=((Topo_cartesian.x.val[1, 1, 1])km, (Topo_cartesian.x.val[end, 1, 1])km),
        y=((Topo_cartesian.y.val[1, 1, 1])km, (Topo_cartesian.y.val[1, end, 1])km),
        z=(-20km, sticky_air*km),
    )

    Grid3D_cart = CartData(xyz_grid(Grid3D.coord1D...));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[30], Phases=[1])

    add_box!(Phases, Temp, Grid3D_cart;
        xlim=(minimum(Grid3D_cart.x.val), maximum(Grid3D_cart.x.val)),
        # ylim=(minimum(Grid3D_cart.y.val), maximum(Grid3D_cart.y.val)),
        zlim=(minimum(Grid3D_cart.z.val), maximum(Grid3D_cart.z.val)),
        phase = lith,
        T = HalfspaceCoolingTemp(Age=20)
    )

    for k in axes(Grid3D_cart.z.val, 3), j in axes(topo_toba, 2), i in axes(topo_toba,1)
        if Grid3D_cart.z.val[i, j, k] > topo_toba[i, j]
            Phases[i, j, k] = 4
        end
    end

    @. Temp[Phases == 4]  = 20
    @. Temp               = max(Temp, 20)
    Grid3D_cart = addfield(Grid3D_cart,(; Phases, Temp))

    ## make a 2D cross section
    Cross_section = cross_section(
        Grid3D_cart;
        dims=(nx, nz),
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )
    Toba_Cross_section = cross_section(
        Topo_cartesian;
        dims=(nx, nz),
        Start=(ustrip(-40.00km), ustrip(50.00km)),
        End=(ustrip(-06.00km), ustrip(20.00km)),
    )

    Cross_section.fields.Phases[above_surface(Cross_section, Topo_cartesian)] .= 4;
    Cross_section.fields.Phases[below_surface(Cross_section, Topo_cartesian)] .= 1;
    Grid2D = create_CartGrid(;
        size=(nx, nz),
        x=(extrema(Cross_section.fields.FlatCrossSection).*km),
        z=(extrema(Cross_section.z.val).*km),
    )
    ## add an ellipsoid
    add_ellipsoid!(Cross_section.fields.Phases, Cross_section.fields.Temp, Cross_section;
        cen    = (mean(Cross_section.x.val), 0, -5.0),
        axes   = (5, 2e3, 2.5),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000),
    )
    ## add a sphere
    add_ellipsoid!(Cross_section.fields.Phases, Cross_section.fields.Temp, Cross_section;
    cen    = (mean(Cross_section.x.val), 0, -5.0),
    axes   = (0.5, 2e3, 0.5),
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1050),
    )
    # write_paraview(Cross_section, "Toba_cross")
    # write_paraview(Grid2D, "2D_Toba_cross")
    li = (Grid2D.L[1].val, Grid2D.L[2].val)
    origin = (Grid2D.min[1].val, Grid2D.min[2].val)

    ph      = Cross_section.fields.Phases[:,:,1]
    T       = Cross_section.fields.Temp[:,:,1]

    ## add topography 2D cross section
    topo1D = Toba_Cross_section.z.val
    return li, origin, ph, T, topo1D
end

function Toba_setup3D(nx,ny,nz; sticky_air=5)
    topo_toba, Topo_cartesian, Lx, Ly = Toba_topo(nx,ny,nz)

    Grid3D = create_CartGrid(;
        size=(nx, ny, nz),
        x=((Topo_cartesian.x.val[1, 1, 1])km, (Topo_cartesian.x.val[end, 1, 1])km),
        y=((Topo_cartesian.y.val[1, 1, 1])km, (Topo_cartesian.y.val[1, end, 1])km),
        z=(-20km, sticky_air*km),
    )

    Grid3D_cart = CartData(xyz_grid(Grid3D.coord1D...));



    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[30], Phases=[1])

    add_box!(Phases, Temp, Grid3D_cart;
        xlim=(minimum(Grid3D_cart.x.val), maximum(Grid3D_cart.x.val)),
        # ylim=(minimum(Grid3D_cart.y.val), maximum(Grid3D_cart.y.val)),
        zlim=(minimum(Grid3D_cart.z.val), maximum(Grid3D_cart.z.val)),
        phase = lith,
        T = HalfspaceCoolingTemp(Age=20)
    )

    # add_volcano!(Phases, Temp, Grid3D_cart;
    #     volcanic_phase  = 1,
    #     center     = (0, 0.0, 0.0),
    #     height     = 3,
    #     radius     = 5,
    #     crater     = 0.5,
    #     base       = 0.0,
    #     # background = topo_toba,
    #     T = HalfspaceCoolingTemp(Age=20)
    # )

    add_ellipsoid!(Phases, Temp, Grid3D_cart;
        cen    = (0, 0, -5.0),
        axes   = (5, 5, 2.5),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000),
    )
    add_sphere!(Phases, Temp, Grid3D_cart;
        cen = (0, 0, -5.0),
        radius = 0.5,
        phase  = ConstantPhase(3),
        T      = ConstantTemp(T=1200)
    )

    for k in axes(Grid3D_cart.z.val, 3), j in axes(topo_toba, 2), i in axes(topo_toba,1)
        if Grid3D_cart.z.val[i, j, k] > topo_toba[i, j]
            Phases[i, j, k] = 4
        end
    end

    @. Temp[Phases == 4]  = 20
    @. Temp               = max(Temp, 20)
    Grid3D_cart = addfield(Grid3D_cart,(; Phases, Temp))
    # write_paraview(Grid3D_cart, "Toba")

    li = (Grid3D.L[1].val, Grid3D.L[2].val, Grid3D.L[3].val)
    origin = (Grid3D.min[1].val, Grid3D.min[2].val, Grid3D.min[3].val)

    ph      = Phases
    T       = Temp

    return li, origin, ph, T
end


function volcano_setup2D(nx,ny,nz;sticky_air=5)
    Lx = Ly = 40
    x = range(0.0, Lx, nx);
    y = range(0.0, Ly, 2);
    z = range(-20, sticky_air, nz);
    Grid = CartData(xyz_grid(x,y,z));


    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, 2, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, 2, nz);

    add_box!(Phases, Temp, Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers=[30], Phases=[1]),
        T = HalfspaceCoolingTemp(Age=20)
    )

    # add_volcano!(Phases, Temp, Grid;
    # volcanic_phase  = 1,
    # center          = (mean(Grid.x.val),  0.0),
    # height          = 4,
    # radius          = 5,
    # crater          = 0.5,
    # base            = 0.0,
    # background      = nothing,
    # T               = HalfspaceCoolingTemp(Age=20)
    # )

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (mean(Grid.x.val), 0,-5.0),
        axes   = (3.5, 2.5, 2.0),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000)
    )
    add_sphere!(Phases, Temp, Grid;
    cen = (mean(Grid.x.val), 0,-5.0),
    radius = 0.5,
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1200)
    )
    # add_cylinder!(Phases, Temp, Grid;
    # base = (mean(Grid.x.val), 0, -3.9),
    # cap  = (mean(Grid.x.val), 0, 0.0),
    # radius = 0.5,
    # phase  = ConstantPhase(2),
    # T      = LinearTemp(Ttop=20, Tbot=1000),
    # )

    Grid = addfield(Grid,(; Phases, Temp))


    li = (abs(last(x)-first(x)), abs(last(z)-first(z)))
    origin = (x[1], z[1])

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:]
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end

function volcano_setup3D(nx,ny,nz;sticky_air=5)
    Lx = Ly = 40

    x = range(0.0, Lx, nx);
    y = range(0.0, Ly, ny);
    z = range(-30, sticky_air, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[15 45], Phases=[1 2])

    add_box!(Phases, Temp, Grid;
        xlim=(-100, 100),
        ylim=(-400, 400.0),
        zlim=(-110.0, 0.0),
        phase = lith,
        T = HalfspaceCoolingTemp(Age=20)
    )

    # add_volcano!(Phases, Temp, Grid;
    #     volcanic_phase  = 1,
    #     center          = (0, 0, 0),
    #     height          = 4,
    #     radius          = 5,
    #     crater          = 0.5,
    #     base            = 0.0,
    #     background      = nothing,
    #     T               = HalfspaceCoolingTemp(Age=20)
    # )

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (0, 0, -5),
        axes   = (2.5, 2.5, 2.5/2),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000)
    )
    add_sphere!(Phases, Temp, Grid;
    cen = (0, 0, -5.0),
    radius = 0.5,
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1200)
    )

    add_cylinder!(Phases, Temp, Grid;
        base = (0, 0, -3.75),
        cap  = (0, 0, 0.0),
        radius = 1.0,
        phase  = ConstantPhase(2),
        T      = LinearTemp(Ttop=20, Tbot=1000)

    )
    # # id = falses(nx, ny, nz)
    # for k in axes(Grid.z.val, 3), j in axes(topo_volcano, 2), i in axes(topo_volcano,1)
    #     # id[i, j, k] = Grid.z.val[i, j, k] > topo_volcano[i, j]
    #     if Grid.z.val[i, j, k] > topo_volcano[i, j]
    #         Phases[i, j, k] = 4
    #     end
    # end
    # surf = Grid.z.val .> 0.0
    # Temp[surf] .= 20.0
    # Phases[surf] .= 4
    @. Temp[Phases == 4]  = 20
    @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))
    # write_paraview(Grid, "Volcano3D")

    li = (abs(last(x)-first(x)),  abs(last(y)-first(y)), abs(last(z)-first(z)))
    origin = (x[1], y[1], z[1])

    ph      = Phases
    T       = Temp

    return li, origin, ph, T
end

function simple_setup_no_FS2D(nx,ny,nz)
    Lx = Ly = 40

    x = range(-Lx / 2, Lx / 2, nx);
    z = range(-20, 0, nz);
    Grid = CartData(xyz_grid(x,0,z));


    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, 1, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, 1, nz);

    add_box!(Phases, Temp, Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        zlim=(minimum(Grid.z.val), maximum(Grid.z.val)),
        phase = LithosphericPhases(Layers=[30], Phases=[1]),
        T = HalfspaceCoolingTemp(Age=20)
    )

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (0, 0,-5),
        axes   = (2.5, 2.5, 2.5/2),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000)
    )
    add_sphere!(Phases, Temp, Grid;
    cen = (0, 0,-5.0),
    radius = 0.5,
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1200)
    )
    add_cylinder!(Phases, Temp, Grid;
    base = (0, 0, -4.0),
    cap  = (0, 0, 4.0),
    radius = 0.5,
    phase  = ConstantPhase(2),
    T      = LinearTemp(Ttop=20, Tbot=1000),
    )

    surf = Grid.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 4
    @. Temp[Phases == 4]  = 20
    @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))


    li = (abs(last(x)-first(x)), abs(last(z)-first(z))).* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases
    T       = Temp
    write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T
end

function simple_setup_no_FS3D(nx,ny,nz)
    Lx = Ly = 40

    x = range(-Lx / 2, Lx / 2, nx);
    y = range(-Ly / 2, Ly / 2, ny);
    z = range(-30, 10, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[15 45], Phases=[1 2])

    add_box!(Phases, Temp, Grid;
        xlim=(-100, 100),
        ylim=(-400, 400.0),
        zlim=(-110.0, 0.0),
        phase = lith,
        T = HalfspaceCoolingTemp(Age=20)
    )

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (0, 0, -5),
        axes   = (2.5, 2.5, 2.5/2),
        phase  = ConstantPhase(2),
        T      = ConstantTemp(T=1000)
    )
    add_sphere!(Phases, Temp, Grid;
    cen = (0, 0, -5.0),
    radius = 0.5,
    phase  = ConstantPhase(3),
    T      = ConstantTemp(T=1200)
    )

    add_cylinder!(Phases, Temp, Grid;
        base = (0, 0, -3.75),
        cap  = (0, 0, 4.0),
        radius = 1.0,
        phase  = ConstantPhase(2),
        T      = LinearTemp(Ttop=20, Tbot=1000)

    )
    # # id = falses(nx, ny, nz)
    # for k in axes(Grid.z.val, 3), j in axes(topo_volcano, 2), i in axes(topo_volcano,1)
    #     # id[i, j, k] = Grid.z.val[i, j, k] > topo_volcano[i, j]
    #     if Grid.z.val[i, j, k] > topo_volcano[i, j]
    #         Phases[i, j, k] = 4
    #     end
    # end

    @. Temp[Phases == 4]  = 20
    @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))
    write_paraview(Grid, "Volcano3D")

    li = (abs(last(x)-first(x)),  abs(last(y)-first(y)), abs(last(z)-first(z)))
    origin = (x[1], y[1], z[1])

    ph      = Phases
    T       = Temp

    return li, origin, ph, T
end
