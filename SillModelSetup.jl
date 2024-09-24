## Model Setup
using GeophysicalModelGenerator

function SillSetup(nx, ny, nz)
    Lx = Ly = 300
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-250, 0.0, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, 2, nz)

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(600+273, nx, 2, nz)

    add_box!(
        Phases,
        Temp,
        Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(-175, -75.0),
        phase=ConstantPhase(2),
        T = ConstantTemp(; T=1000+273),
    )

    # add_sphere!(Phases, Temp, Grid;
    # cen = (mean(Grid.x.val), 0,-90.0),
    # radius = 5,
    # phase  = ConstantPhase(3),
    # T      = ConstantTemp(T=1100+273)
    # )

    Grid = addfield(Grid, (; Phases, Temp))

    li = (abs(last(x) - first(x)), abs(last(z) - first(z)))
    origin = (x[1], z[1])

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :]

    return li, origin, ph, T, Grid
end
