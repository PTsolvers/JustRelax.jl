using LinearAlgebra#, GLMakie
using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 1)
environment!(model)

# model resolution (number of gridpoints)
nx = 64

@parallel_indices (i) function init_T!(T, z)
    if z[i] == maximum(z)
        T[i] = 300.0
    elseif z[i] == minimum(z)
        T[i] = 3500.0
    else
        T[i] = z[i] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

@views function diffusion_1D()
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr        # total simulation time
    dt = 50 * kyr        # physical time step

    # Physical domain
    lx, ly = 100e3, 100e3    # domain size
    ni = (nx,)
    li = (lx,)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li; origin=(-ly,)) # nodes at the center and vertices of the cells

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    ρ = @fill(3300.0, ni...)
    Cp = @fill(1.2e3, ni...)
    K = @fill(3.0, ni...)
    ρCp = @. Cp * ρ
    thermal_parameters = ThermalParameters(K, ρCp)

    # Numerics
    # nx, ny  = 2*256, 2*256  # numerical grid resolution
    tol = 1e-8          # tolerance
    itMax = 1e3           # max number of iterations
    nout = 10            # tol check
    CFL = 0.5 / sqrt(2)     # CFL number

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = (flux_x=false, flux_y=false)

    @parallel (1:nx, 1:ny) init_T!(thermal.T, xci[2])
    @parallel assign!(thermal.Told, thermal.T)

    t = 0.0
    it = 0
    ittot = 0
    nt = Int(ceil(ttot / dt))
    # Physical time loop
    while it < nt
        solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
            dt;
            iterMax=10e3,
            nout=10,
            verbose=false,
        )
        it += 1
    end

    return nothing
end
