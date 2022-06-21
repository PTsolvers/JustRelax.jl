@parallel_indices (i) function init_T!(T, z)
    if i == length(T)
        T[i] = 300.0
    elseif i == 1
        T[i] = 3500.0
    else
        T[i] = z[i] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function diffusion_1D(; nx=128, lx=100e3, ρ=3.3e3, Cp=1.2e3, K=3.0)
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx,)
    li = (lx,)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    xci, = lazy_grid(di, li; origin=(-lx,)) # nodes at the center and vertices of the cells

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    ρ = @fill(ρ, ni...)
    Cp = @fill(Cp, ni...)
    K = @fill(K, ni...)
    ρCp = @. Cp * ρ
    thermal_parameters = ThermalParameters(K, ρCp)

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; CFL=0.8 / √3)

    @parallel (1:nx) init_T!(thermal.T, xci[1])
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
            ni,
            di,
            dt;
            iterMax=10e3,
            nout=1,
            verbose=false,
        )
        it += 1
        t += dt
    end

    return (ni=ni, xci=xci, li=li, di=di), thermal, iters
end
