using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny)
    model_depth = 700
    # Our starting basis is the example above with ridge and overriding slab
    nx, nz = nx, ny
    Tbot = 1474.0
    x = range(0, 3000, nx)
    air_thickness = 50.0
    z = range(-model_depth, air_thickness, nz)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx, 1, nz)
    Temp = fill(Tbot, nx, 1, nz)
    Tlab = 1300

    # phases
    # 1: asthenosphere
    # 2: lithosphere
    # 3: air
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (0, 3000),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [], Phases = [0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add oceanic plate
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (1000, 3000),
        zlim = (-100, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [100], Phases = [1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add slab
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (1000, 1200),
        zlim = (-100, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 90,
        phase = LithosphericPhases(Layers = [200], Phases = [1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    Phases .+= 1
    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 3

    Grid2D = addfield(Grid2D, (; Phases, Temp))

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :]

    return li, origin, ph, T
end
