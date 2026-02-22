## Model Setup
using GeophysicalModelGenerator

function setup2D(
        nx, nz;
        sticky_air = 5.0e0,
        dimensions = (30.0e0, 20.0e0), # extent in x and y in km
        flat = true,
        chimney = false,
        volcano_size = (3.0e0, 5.0e0),
        conduit_radius = 0.2,
        chamber_T = 1.0e3,
        chamber_depth = 5.0e0,
        chamber_radius = 2.0e0,
        aspect_x = 1.5,
    )

    Lx = Ly = dimensions[1]
    x = range(0.0, Lx, nx)
    y = range(0.0, Ly, 2)
    z = range(-dimensions[2], sticky_air, nz)
    Grid = CartData(xyz_grid(x, y, z))

    # Allocate Phase and Temp arrays
    Phases = fill(6, nx, 2, nz)
    Temp = fill(0.0, nx, 2, nz)

    add_box!(
        Phases, Temp, Grid;
        xlim = (minimum(Grid.x.val), maximum(Grid.x.val)),
        ylim = (minimum(Grid.y.val), maximum(Grid.y.val)),
        zlim = (minimum(Grid.z.val), 0.0),
        phase = LithosphericPhases(Layers = [chamber_depth], Phases = [1, 2]),
        T = HalfspaceCoolingTemp(Age = 20)
    )

    if !flat
        add_volcano!(
            Phases, Temp, Grid;
            volcanic_phase = 1,
            center = (mean(Grid.x.val), 0.0),
            height = volcano_size[1],
            radius = volcano_size[2],
            crater = 0.5,
            base = 0.0,
            background = nothing,
            T = HalfspaceCoolingTemp(Age = 20)
        )
    end

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -chamber_depth),
        axes = (chamber_radius * aspect_x, 2.5, chamber_radius),
        phase = ConstantPhase(3),
        T = ConstantTemp(T = chamber_T - 100.0e0)
    )

    add_ellipsoid!(
        Phases, Temp, Grid;
        cen = (mean(Grid.x.val), 0, -(chamber_depth - (chamber_radius / 2))),
        axes = ((chamber_radius / 1.25) * aspect_x, 2.5, (chamber_radius / 2)),
        phase = ConstantPhase(4),
        T = ConstantTemp(T = chamber_T)
    )

    # add_sphere!(Phases, Temp, Grid;
    #     cen    = (mean(Grid.x.val), 0, -(chamber_depth-(chamber_radius/2))),
    #     radius = (chamber_radius/2),
    #     phase  = ConstantPhase(4),
    #     T      = ConstantTemp(T=chamber_T+100)
    # )

    if chimney
        add_cylinder!(
            Phases, Temp, Grid;
            base = (mean(Grid.x.val), 0, -(chamber_depth - chamber_radius)),
            cap = (mean(Grid.x.val), 0, flat ? 0.0e0 : volcano_size[1]),
            radius = conduit_radius,
            phase = ConstantPhase(5),
            T = ConstantTemp(T = chamber_T),
        )
    end

    Grid = addfield(Grid, (; Phases, Temp))
    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :] .+ 273
    V_total = 4 / 3 * π * (chamber_radius * aspect_x) * chamber_radius * (chamber_radius * aspect_x)
    V_erupt = 4 / 3 * π * (chamber_radius / 1.25) * aspect_x * (chamber_radius / 2) * ((chamber_radius / 1.25) * aspect_x)
    R = ((chamber_depth - chamber_radius)) / (chamber_radius * aspect_x)
    chamber_diameter = 2 * (chamber_radius * aspect_x)
    chamber_erupt = 2 * ((chamber_radius / 1.25) * aspect_x)
    printstyled("Magma volume of the initial chamber: $(round(V_total; digits = 3)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Eruptible magma volume: $(round(V_erupt; digits = 3)) km³ \n"; bold = true, color = :red, blink = true)
    printstyled("Roof ratio (Depth/half-axis width): $R \n"; bold = true, color = :cyan)
    printstyled("Chamber diameter: $chamber_diameter km \n"; bold = true, color = :light_yellow)
    printstyled("Eruptible chamber diameter: $chamber_erupt km \n"; bold = true, color = :light_yellow)
    # write_paraview(Grid, "Volcano2D")
    return li, origin, ph, T, Grid
end
