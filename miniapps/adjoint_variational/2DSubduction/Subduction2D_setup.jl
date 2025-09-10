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
    
    # right lithopshere
    lith_right = LithosphericPhases(Layers=[35,80], Phases=[4,2,0], Tlab=1350)
    add_box!(Phases, Temp, Grid2D; xlim=(2000.0,3000.0), ylim=(-400, 400.0), zlim=(-800.0, 0.0), phase = lith_right,
        Origin=(-0,0,0), T=HalfspaceCoolingTemp(Age=120,Adiabat=0.4), StrikeAngle=0)
    
    # left lithosphere
    lith_left = LithosphericPhases(Layers=[15,80], Phases=[5,1,0], Tlab=1350)
    add_box!(Phases, Temp, Grid2D; xlim=(0.0,2000.0), ylim=(-400, 400.0), zlim=(-800.0, 0.0), phase = lith_left,
        Origin=(-0,0,0),
        T=SpreadingRateTemp(SpreadingVel=1.0, MORside="left",Adiabat=0.4,maxAge=120), StrikeAngle=0)
     
        
    trench= Trench(Start = (2000.0,-400.0), End = (2000.0,400.0), θ_max = 60.0, direction = -1.0, n_seg = 100, Length = 400.0, Thickness = 160.0, Lb = 300.0, d_decoupling = 1000.0, type_bending =:Ribe,WeakzoneThickness=5,WeakzonePhase=5)
    T_slab = LinearWeightedTemperature(F1=HalfspaceCoolingTemp(Age=120,Adiabat=0.4), F2=McKenzie_subducting_slab(Tsurface=0,v_cm_yr=3, Adiabat = 0.4),crit_dist=400)
    add_slab!(Phases, Temp, Grid2D, trench, phase = lith_left,T=T_slab)
    

    Phases .+= 1
    surf = Grid2D.z.val .> 0.0
    Temp[surf] .= 20.0
    Phases[surf] .= 7

    Grid2D = addfield(Grid2D, (; Phases, Temp))

    li = (abs(last(x) - first(x)), abs(last(z) - first(z))) .* 1.0e3
    origin = (x[1], z[1]) .* 1.0e3

    ph = Phases[:, 1, :]
    T = Temp[:, 1, :]

    return li, origin, ph, T
end
