using GeophysicalModelGenerator

function GMG_subduction_2D(model_depth, xvi, nx, ny)
    nx, nz = nx, ny
    Tbot = 1474.0
    x = range(minimum(xvi[1]), maximum(xvi[1]), nx)
    z = range(minimum(xvi[2]), maximum(xvi[2]), nz)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx, 1, nz)
    Temp = fill(Tbot, nx, 1, nz)
    Tlab = 1300
    # lith   = LithosphericPhases(Layers=[80], Phases=[1 0], Tlab=Tlab)

    # phases
    # 0: asthenosphere
    # 1: lithosphere
    # 2: subduction lithosphere
    # 3: oceanic crust
    # 4: air
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

    # Add left oceanic plate
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (100, 2900),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [80], Phases = [1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add right oceanic plate crust
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (1570, 2800),
        zlim = (-model_depth, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = 0,
        phase = LithosphericPhases(Layers = [8 72], Phases = [2 1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    # Add slab
    add_box!(
        Phases,
        Temp,
        Grid2D;
        xlim = (1570, 1320),
        zlim = (-80, 0.0),
        Origin = nothing, StrikeAngle = 0, DipAngle = -30,
        phase = LithosphericPhases(Layers = [8 80], Phases = [2 1 0], Tlab = Tlab),
        T = HalfspaceCoolingTemp(Tsurface = 20, Tmantle = Tbot, Age = 50, Adiabat = 0)
    )

    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 3


    Grid2D = addfield(Grid2D, (; Phases, Temp))
    # Which looks like
    write_paraview(Grid2D, "Initial_Setup_Subduction_$(igg.me)")

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :] .+ 1
    T = Temp[:, 1, :]

    return li, origin, ph, T
end
