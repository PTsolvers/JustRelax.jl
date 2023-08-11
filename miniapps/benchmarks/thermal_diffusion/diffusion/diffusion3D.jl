using MPI

@parallel_indices (i, j, k) function init_T!(T, z)
    if z[k] == maximum(z)
        T[i, j, k] = 300.0
    elseif z[k] == minimum(z)
        T[i, j, k] = 3500.0
    else
        T[i, j, k] = z[k] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function diffusion_3D(;
    nx=32,
    ny=32,
    nz=32,
    lx=100e3,
    ly=100e3,
    lz=100e3,
    ρ0=3.3e3,
    Cp0=1.2e3,
    K0=3.0,
    init_MPI=JustRelax.MPI.Initialized() ? false : true,
    finalize_MPI=false,
)
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx, ny, nz)
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=(0, 0, -lz)) # nodes at the center and vertices of the cells
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=0.0, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=Cp0),
        Conductivity      = ConstantConductivity(; k=K0),
    )

    # fields needed to compute density on the fly
    P = @zeros(ni...)
    args = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    # Boundary conditions
    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false, front=true, back=true), 
        periodicity = (left = false, right = false, top = false, bot = false, front=false, back=false),
    )

    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[3])

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Physical time loop
    local iters
    while it < nt
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di,;
            igg
        )
       
        t += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, li=li, di=di), thermal
end
