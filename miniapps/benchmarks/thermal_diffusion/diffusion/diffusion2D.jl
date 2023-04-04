@parallel_indices (i, j) function init_T!(T, z)
    if z[j] == maximum(z)
        T[i, j] = 300.0
    elseif z[j] == minimum(z)
        T[i, j] = 3500.0
    else
        T[i, j] = z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function diffusion_2D(;
    nx = 32,
    ny = 32,
    lx = 100e3,
    ly = 100e3,
    ρ = 3.3e3,
    Cp = 1.2e3,
    K = 3.0,
)
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    xci, = lazy_grid(di, li; origin = (0, -ly)) # nodes at the center and vertices of the cells

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    ρ = @fill(ρ, ni...)
    Cp = @fill(Cp, ni...)
    K = @fill(K, ni...)
    ρCp = @. Cp * ρ
    thermal_parameters = ThermalParameters(K, ρCp)

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = (flux_x = false, flux_y = false)

    @parallel (1:nx, 1:ny) init_T!(thermal.T, xci[2])
    @parallel assign!(thermal.Told, thermal.T)

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))
    # Physical time loop
    local iters
    while it < nt
        iters = solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
            dt;
            iterMax = 10e3,
            nout = 1,
            verbose = false,
        )
        it += 1
        t += dt
    end

    return (ni = ni, xci = xci, li = li, di = di), thermal, iters
end
