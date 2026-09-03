using GeophysicalModelGenerator

function GMG_subduction_2D(nx, ny; Tlab = 1300.0, v_spread_cm_yr = 2.0, AgeRidge = 40.0, maxAge = 80.0)
    model_depth = 660
    nx, nz = nx, ny
    Tbot = 1474.0
    x = range(-500, 3500, nx)
    air_thickness = 30.0
    z = range(-model_depth, air_thickness, nz)
    Grid2D = CartData(xyz_grid(x, 0, z))
    Phases = zeros(Int64, nx, 1, nz)
    Temp = fill(Tbot, nx, 1, nz)

    # phases (before the +1 shift at the end)
    # 0: asthenosphere, 1: left mantle lithosphere, 2: right mantle lithosphere,
    # 3: left crust, 4: right crust, 5: weak decoupling zone, 6: marker layer
    #
    # `LithosphericPhases` is used without `Tlab`: it would set *every* layer, crust
    # included, to asthenosphere wherever T > Tlab, which strips the ridge of any
    # material. The lithosphere-asthenosphere boundary is cut at the end instead, on
    # the mantle-lithosphere phases only.

    add_box!(
        Phases, Temp, Grid2D; xlim = (-500.0, 3500.0), ylim = (-400, 400.0), zlim = (-800.0, 0.0), phase = ConstantPhase(0),
        Origin = (-0, 0, 0), T = HalfspaceCoolingTemp(Age = 0.0, Adiabat = 0.0), StrikeAngle = 0
    )

    # Overriding plate, defined twice with different thickness to accommodate the
    # bending zone in front of the trench
    lith_right = LithosphericPhases(Layers = [30, 80], Phases = [4, 2, 0])
    add_box!(
        Phases, Temp, Grid2D; xlim = (2000.0, 3500.0), ylim = (-400, 400.0), zlim = (-800.0, 0.0), phase = lith_right,
        Origin = (-0, 0, 0), T = HalfspaceCoolingTemp(Age = 140, Adiabat = 0.0), StrikeAngle = 0
    )
    add_box!(
        Phases, Temp, Grid2D; xlim = (1850.0, 2000.0), ylim = (-400, 400.0), zlim = (-80.0, 0.0), phase = lith_right,
        Origin = (-0, 0, 0), T = HalfspaceCoolingTemp(Age = 150, Adiabat = 0.0), StrikeAngle = 0
    )

    # Subducting oceanic plate, accreted at a mid-ocean ridge on the left. The thermal
    # age grows with distance from the ridge and saturates at `maxAge`, so the plate
    # reaches the trench with the same age the slab is built with. `AgeRidge` offsets
    # the age at the boundary itself: the model is not kinematically driven, so a
    # zero-age boundary would have no lithosphere left after the `Tlab` cut below and
    # would neck under slab pull. It must stay old enough to carry a full plate.
    lith_left = LithosphericPhases(Layers = [15, 80], Phases = [3, 1, 0])

    #=
    # Old oceanic plate left of the ridge, mirroring the ridge structure with maxAge
    add_box!(
        Phases, Temp, Grid2D; xlim = (-1000.0, -500.0), ylim = (-400, 400.0), zlim = (-800.0, 0.0), phase = lith_left,
        Origin = (-0, 0, 0), T = SpreadingRateTemp(SpreadingVel = v_spread_cm_yr, MORside = "right", Adiabat = 0.4, AgeRidge = AgeRidge, maxAge = maxAge), StrikeAngle = 0
    )

    add_box!(
        Phases, Temp, Grid2D; xlim = (-500.0, 1850.0), ylim = (-400, 400.0), zlim = (-800.0, 0.0), phase = lith_left,
        Origin = (-0, 0, 0),
        T = SpreadingRateTemp(SpreadingVel = v_spread_cm_yr, MORside = "left", Adiabat = 0.4, AgeRidge = AgeRidge, maxAge = maxAge), StrikeAngle = 0
    )
=#
    add_box!(
        Phases, Temp, Grid2D; xlim = (-500.0, 1850.0), ylim = (-400, 400.0), zlim = (-800.0, 0.0), phase = lith_left,
        Origin = (-0, 0, 0),
        T = SpreadingRateTemp(SpreadingVel = v_spread_cm_yr, MORside = "left", Adiabat = 0.0, AgeRidge = AgeRidge, maxAge = maxAge), StrikeAngle = 0
    )

    # `WeakzonePhase = 5` becomes phase 6 after the shift below: the weak decoupling
    # channel between the subducting and the overriding plate
    trench = Trench(Start = (1800.0, -400.0), End = (1800.0, 400.0), θ_max = 30.0, direction = -1.0, n_seg = 200, Length = 500.0, Thickness = 180.0, Lb = 500.0, d_decoupling = 500.0, type_bending = :Ribe, WeakzoneThickness = 10, WeakzonePhase = 5)
    T_slab = LinearWeightedTemperature(F1 = HalfspaceCoolingTemp(Age = maxAge, Adiabat = 0.0), F2 = McKenzie_subducting_slab(Tsurface = 20, v_cm_yr = 2.0, Adiabat = 0.0), crit_dist = 2000)
    add_slab!(Phases, Temp, Grid2D, trench, phase = lith_left, T = T_slab)

    # overriding plate observation box
    # add_box!(
    #     Phases, Temp, Grid2D; xlim = (1980.0, 2000.0), ylim = (-400, 400.0), zlim = (-20.0, -2.0), phase = ConstantPhase(6),
    #     Origin = (-0, 0, 0), StrikeAngle = 0
    # )

    # Lithosphere-asthenosphere boundary, tracking the `Tlab` isotherm. Only the
    # mantle-lithosphere phases are converted, so the plate thins towards the ridge
    # while the freshly accreted crust is kept.
    Phases[findall(Temp .> Tlab .&& (Phases .== 1 .|| Phases .== 2))] .= 0

    Phases .+= 1
    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 8

    Grid2D = addfield(Grid2D, (; Phases, Temp))
    write_paraview(Grid2D,"Grid2D_SubductionCurvedOverriding");

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :]

    return li, origin, ph, T .+ 273
end
