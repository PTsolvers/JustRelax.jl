using GLMakie
using GeophysicalModelGenerator, GMT
using Interpolations

function Khefalonia_topo(N)
    # import topo from GMT's server
    Topo = import_topo([20.2, 20.8, 38.3, 38.5], file="@earth_relief_03s")
    # Topo = import_topo([20, 21, 38.3, 38.5], file="@earth_relief_03s")
    # extract 2D array with the topographic elevation
    surf = Topo.depth.val[:,:,1]
    # l = maximum(abs.(Topo.depth.val))
    # heatmap(surf, colorrange=(-l,l), colormap=:oleron )
    # # GLMakie.surface(surf, colorrange=(-l,l), colormap=:oleron )

    # our model is in [km] -> need to convert from lat-long
    x = LinRange(20.4, 20.8, N) |> collect
    y = LinRange(38.3, 38.5, N) |> collect
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
    
    return surf_interp, Lx, Ly, surf_interp[:, 30]
end

function Khefalonia_setup(Nx, Nz)
    surf, Lx, Ly, topo = Khefalonia_topo(Nx)

    nx = Nx 
    nz = Nz
    ny = 10
    x = range(-Lx / 2, Lx / 2, nx);
    y = range(-eps(), eps(), ny);
    z = range(-18, 5, nz);
    # z = range(-14.5, 5, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[20 45 100], Phases=[1 2 3])

    add_box!(Phases, Temp, Grid; 
        xlim  =(-Lx, Lx), 
        ylim  =(-Ly, Ly), 
        zlim  =(-60.0, 0.0), 
        phase = lith, 
        T     = HalfspaceCoolingTemp(Age=20)
    )
   

    add_ellipsoid!(Phases, Temp, Grid; 
        cen    = (0, 0, -6), 
        axes   = (0.75, 1, 0.75),
        phase  = ConstantPhase(3),
        # T  = ConstantTemp(150),
    )

    # id = falses(nx, ny, nz)
    for k in axes(Grid.z.val, 3), j in axes(topo, 2), i in axes(topo,1)
        # id[i, j, k] = Grid.z.val[i, j, k] > topo_volcano[i, j]
        if Grid.z.val[i, j, k] > topo[i]
            if Grid.z.val[i, j, k]> 0 
                Phases[i, j, k] = 4 # air
            else
                Phases[i, j, k] = 5 # water
            end
            Temp[i, j, k] = 20
        end
    end

    # @. Temp[Phases == 3]   = 20
    # @. Temp[Phases == 4]   = 20 
    # @. Temp[Phases == 10]   += 100
    # @. Phases[Phases == 10]  = 1 
    # @. Temp               = max(Temp, 20)
    @views @. Temp[Phases == 3]   += 50

    Grid = addfield(Grid,(; Phases, Temp))

    li     = (abs(last(x)-first(x)),  abs(last(z)-first(z))) .* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases
    T       = Temp

    return li, origin, ph[:,1,:], T[:,1,:] .+ 273
end


function flat_setup(Nx, Nz)

    Lx = Ly = 30
    nx = Nx 
    nz = Nz
    ny = 10
    x = range(-Lx / 2, Lx / 2, nx);
    y = range(-eps(), eps(), ny);
    z = range(-18, 4, nz);
    # z = range(-14.5, 5, nz);
    Grid = CartData(xyz_grid(x,y,z));

    # Now we create an integer array that will hold the `Phases` information (which usually refers to the material or rock type in the simulation)
    Phases = fill(1, nx, ny, nz);

    # In many (geodynamic) models, one also has to define the temperature, so lets define it as well
    Temp = fill(20.0, nx, ny, nz);

    lith = LithosphericPhases(Layers=[20 45 100], Phases=[1 2 3])

    add_box!(Phases, Temp, Grid; 
        xlim  =(-Lx, Lx), 
        ylim  =(-Ly, Ly), 
        zlim  =(-60.0, 0.0), 
        phase = lith, 
        T     = HalfspaceCoolingTemp(Age=20)
    )
   
    add_ellipsoid!(Phases, Temp, Grid; 
        cen    = (0, 0, -2), 
        axes   = (1, 1, 1) .* 0.5,
        phase  = ConstantPhase(3),
        # T  = ConstantTemp(150),
    )

    for I in eachindex(Grid.z.val)
        if Grid.z.val[I...] > 0 
            Phases[I...] = 4 # air
            Temp[I...] = 20
        end
    end

    # @. Temp[Phases == 3]   = 20
    # @. Temp[Phases == 4]   = 20 
    @views @. Temp[Phases == 3]   += 50
    # @. Phases[Phases == 3]  = 1
    # @. Temp               = max(Temp, 20)
    Grid = addfield(Grid,(; Phases, Temp))

    li     = (abs(last(x)-first(x)),  abs(last(z)-first(z))) .* 1e3
    origin = (x[1], z[1]) .* 1e3

    ph      = Phases
    T       = Temp

    return li, origin, ph[:,1,:], T[:,1,:] .+ 273
end