## Model Setup
using GeophysicalModelGenerator

function setup2D(
    nx, nz;
    sticky_air     = 5e0,
    dimensions     = (30e0, 20e0), # extent in x and y in km
    flat           = true,
    chimney        = false,
    volcano_size   = (3e0, 5e0),
    conduit_radius = 0.25,
    chamber_T      = 1e3,
    chamber_depth  = 5e0,
    chamber_radius = 2e0,
    aspect_x       = 1.5,
)

    Lx = Ly = dimensions[1]
    x = range(0.0, Lx, nx);
    y = range(0.0, Ly, 2);
    z = range(-dimensions[2], sticky_air, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(6, nx, 2, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(0.0, nx, 2, nz);

    add_box!(Phases, Temp, Grid;
        xlim=(minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim=(minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim=(minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers=[chamber_depth], Phases=[1, 2]),
        T = HalfspaceCoolingTemp(Age=20)
    )

    if !flat
        add_volcano!(Phases, Temp, Grid;
            volcanic_phase  = 1,
            center          = (mean(Grid.x.val),  0.0),
            height          = volcano_size[1],
            radius          = volcano_size[2],
            crater          = 0.5,
            base            = 0.0,
            background      = nothing,
            T               = HalfspaceCoolingTemp(Age=20)
        )
    end

    add_ellipsoid!(Phases, Temp, Grid;
        cen    = (mean(Grid.x.val), 0, -chamber_depth),
        axes   = (chamber_radius * aspect_x, 2.5, chamber_radius),
        phase  = ConstantPhase(3),
        T      = ConstantTemp(T=chamber_T)
    )

    # add_sphere!(Phases, Temp, Grid;
    #     cen    = (mean(Grid.x.val), 0, -(chamber_depth-(chamber_radius/2))),
    #     radius = (chamber_radius/2),
    #     phase  = ConstantPhase(4),
    #     T      = ConstantTemp(T=chamber_T+100)
    # )

    if chimney
        add_cylinder!(Phases, Temp, Grid;
            base = (mean(Grid.x.val), 0, -(chamber_depth-chamber_radius)),
            cap  = (mean(Grid.x.val), 0, flat ? 0e0 : volcano_size[1]),
            radius = conduit_radius,
            phase  = ConstantPhase(5),
            # T      = LinearTemp(Ttop=20, Tbot=1000),
            # T      = ConstantTemp(T=800),
            T      = ConstantTemp(T=chamber_T),
        )
    end

    Grid = addfield(Grid,(; Phases, Temp))
    li = (abs(last(x)-first(x)), abs(last(z)-first(z))) .* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases[:,1,:]
    T       = Temp[:,1,:] .+ 273
    V = 4/3 * π * (chamber_radius*aspect_x) * chamber_radius * 1.0
    printstyled("Magma volume of the initial chamber: $(round(V; digits=3)) km³ \n"; bold=true, color=:red)
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end
