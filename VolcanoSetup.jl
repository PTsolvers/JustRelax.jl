using GLMakie
using GeophysicalModelGenerator, GMT
using Interpolations

function Etna_topo(N)
    Topo = import_topo([14.5, 15.5, 37.2, 38.2], file="@earth_relief_01m")
    surf = Topo.depth.val[:,:,1]

    l = maximum(abs.(Topo.depth.val))
    # heatmap(Topo.depth.val[:,:,1], colormap=:oleron, colorrange= (-l,l))
    # GLMakie.surface(surf, colormap=:oleron, colorrange= (-l,l))

    ##
    x = LinRange(14.5, 15.5, N) |> collect
    y = LinRange(37.2, 38.2, N) |> collect

    X = (
        Topo.lon.val[:, 1, 1],
        Topo.lat.val[1, :, 1],
    )

    itp = interpolate(X, surf, Gridded(Linear()))
    surf_interp = [itp(x, y) for x in x, y in y]

    # GLMakie.surface(surf_interp, colormap=:oleron, colorrange= (-l,l))

    lat_dist  = extrema(X[1]) |> collect |> diff |> first |> abs
    long_dist = extrema(X[2]) |> collect |> diff |> first |> abs
    Ly        = lat_dist * 110.574
    Lx        = long_dist * 111.320*cos(lat_dist)
    
    return surf_interp, Lx, Ly
end

function volcano_setup(N)
    topo_volcano, Lx, Ly = Etna_topo(N)

    nx = ny = nz = N
    x = range(-Lx / 2, Lx / 2, nx);
    y = range(-Ly / 2, Ly / 2, ny);
    z = range(-50, 10, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(4, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(1350.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[15 45 100], Phases=[1 2 3])

    add_box!(Phases, Temp, Grid; 
        xlim=(-100, 100), 
        ylim=(-400, 400.0), 
        zlim=(-110.0, 0.0), 
        phase = lith, 
        T = HalfspaceCoolingTemp(Age=20)
    )
        
    id = falses(nx, ny, nz)
    for k in axes(topo_volcano,3), j in axes(topo_volcano, 2), i in axes(topo_volcano,1)
        id[i, j, k] = Grid.z.val[i, j, k] > topo_volcano[i, j]
    end

    # add_volcano!(Phases, Temp, Grid;
    #     volcanic_phase  = 1,
    #     center     = (0, 0, 0),
    #     height     = 4,
    #     radius     = 10,
    #     crater     = 0.0,
    #     base       = 0.0,
    #     background = nothing,
    #     T = HalfspaceCoolingTemp(Age=20)
    # )

    add_ellipsoid!(Phases, Temp, Grid; 
        cen    = (0, 0, -12.5), 
        axes   = (5, 5, 2.5),
        phase  = ConstantPhase(3),
    )

    @. Temp[Phases == 3] += 500
    @. Temp[Phases == 4]  = 20 
    @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))

    li = (abs(last(x)-first(x)),  abs(last(y)-first(y)), abs(last(z)-first(z)))
    origin = (x[1], y[1], z[1])

    ph      = Phases
    T       = Temp

    return li, origin, ph, T
end