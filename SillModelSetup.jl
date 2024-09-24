## Model Setup

function SillSetup(nx, ny, nz; sticky_air=5)
    Lx = Ly = 50
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-25, sticky_air, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, 2, nz)

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, 2, nz)

    add_box!(
        Phases,
        Temp,
        Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(minimum(Grid.z.val), 0.0),
        phase=LithosphericPhases(; Layers=[30], Phases=[1]),
        T=HalfspaceCoolingTemp(; Age=20),
    )

    # add_volcano!(Phases, Temp, Grid;
    # volcanic_phase  = 1,
    # center          = (mean(Grid.x.val),  0.0),
    # height          = 3,
    # radius          = 5,
    # crater          = 0.5,
    # base            = 0.0,
    # background      = nothing,
    # T               = HalfspaceCoolingTemp(Age=20)
    # )

    add_ellipsoid!(
        Phases,
        Temp,
        Grid;
        cen=(mean(Grid.x.val), 0, -5.0),
        axes=(3.5, 2.5, 2.0),
        phase=ConstantPhase(2),
        T=ConstantTemp(; T=1000),
    )
    # add_sphere!(Phases, Temp, Grid;
    # cen = (mean(Grid.x.val), 0,-5.0),
    # radius = 0.5,
    # phase  = ConstantPhase(3),
    # T      = ConstantTemp(T=1100)
    # )
    # add_cylinder!(Phases, Temp, Grid;
    # base = (mean(Grid.x.val), 0, -3.25),
    # cap  = (mean(Grid.x.val), 0, 3.00),
    # radius = 0.05,
    # phase  = ConstantPhase(2),
    # # T      = LinearTemp(Ttop=20, Tbot=1000),
    # # T      = ConstantTemp(T=800),
    # T      = ConstantTemp(T=1000),
    # )

    Grid = addfield(Grid, (; Phases, Temp))

    li = (abs(last(x) - first(x)), abs(last(z) - first(z)))
    origin = (x[1], z[1])

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :]
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end

function simple_setup_no_FS2D(nx, ny, nz)
    Lx = Ly = 40

    x = range(-Lx / 2, Lx / 2, nx)
    z = range(-20, 0, nz)
    Grid = CartData(xyz_grid(x, 0, z))

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, 1, nz)

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, 1, nz)

    add_box!(
        Phases,
        Temp,
        Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        zlim=(minimum(Grid.z.val), maximum(Grid.z.val)),
        phase=LithosphericPhases(; Layers=[30], Phases=[1]),
        T=HalfspaceCoolingTemp(; Age=20),
    )

    add_ellipsoid!(
        Phases,
        Temp,
        Grid;
        cen=(0, 0, -5),
        axes=(2.5, 2.5, 2.5 / 2),
        phase=ConstantPhase(2),
        T=ConstantTemp(; T=1000),
    )
    add_sphere!(
        Phases,
        Temp,
        Grid;
        cen=(0, 0, -5.0),
        radius=0.5,
        phase=ConstantPhase(3),
        T=ConstantTemp(; T=1200),
    )
    add_cylinder!(
        Phases,
        Temp,
        Grid;
        base=(0, 0, -4.0),
        cap=(0, 0, 4.0),
        radius=0.5,
        phase=ConstantPhase(2),
        T=LinearTemp(; Ttop=20, Tbot=1000),
    )

    surf = Grid.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 4
    @. Temp[Phases == 4] = 20
    @. Temp = max(Temp, 20)
    Grid = addfield(Grid, (; Phases, Temp))

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph = Phases
    T = Temp
    write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T
end
