## HELPER FUNCTIONS

if model.device == :cpu
    @eval(PTtype(::Type{Array}) = Array)
else
    @eval(PTtype(::Type{CuArray}) = CuArray)
end

@parallel_indices (i, j, k) function init_T!(T, z)
    if z[k] == maximum(z)
        T[i, j, k] = 300.0
    elseif z[k] == minimum(z)
        T[i, j, k] = 3500.0
    else
        T[i, j, k] = z[k] * (1900 - 1600) / minimum(z) + 1600
    end
    return nothing
end

## MAIN FUNCTION 

Myr = 1e6 * 3600 * 24 * 365.25
kyr = 1e3 * 3600 * 24 * 365.25
ttot = 1 * Myr        # total simulation time
dt = 50 * kyr        # physical time step
nx = 64
ny = 64
nz = 64
lx = 1000e3
ly = 1000e3
lz = 1000e3
b_width = (4, 4, 4)
init_MPI = MPI.Initialized() ? false : true
finalize_MPI = false

function DiffusionSlab(;
    ttot=150e6 * 3600 * 24 * 365,
    dt=500e3 * 3600 * 24 * 365,
    nx::Integer=32 - 1,
    ny::Integer=32 - 1,
    nz::Integer=32 - 1,
    lx=1000e3,
    ly=1000e3,
    lz=1000e3,
    b_width::NTuple{3,Integer}=(4, 4, 4),
    init_MPI=MPI.Initialized() ? false : true,
    finalize_MPI=false,
)

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI, select_device=false)...) # init MPI
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li; origin=(0, 0, -lz)) # nodes at the center and vertices of the cells

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)
    @parallel init_T!(thermal.T, xci[3])
    T0 = deepcopy(thermal.T)
    @parallel assign!(thermal.Told, thermal.T)

    # general numerical coeffs for PT solver
    pt_thermal = PTThermalCoeffs(di, li; Resc=6π, CFL=0.5 / √3, ϵ=1e-6)

    ## Density
    ρm = 3.3e3 # density of the mantle
    ρ = @fill(ρm, ni...)

    ## Temperature
    κm = 3.3
    Cpm = 1.2e3
    ρCpm = @. Cpm * ρm
    ρCp = @fill(ρCpm, ni...)
    κ = @fill(κm, ni...)
    thermal_parameters = ThermalParameters(κ, ρCp)

    ## Boundary conditions
    # thermal_bc = (flux_x=false, flux_y=true, flux_z=false)
    thermal_bc = (flux_x=false, flux_y=false, flux_z=false)

    ## Time loop
    t = 0.0
    println("Starting solver")

    while t < ttot
        tic()
        solve!(
            thermal, pt_thermal, thermal_parameters, thermal_bc, ni, di, igg, dt; nout=100
        )
        t_toc = toc()
        println("Done in $(t_toc) s")

        t += dt
    end

    finalize_global_grid(; finalize_MPI=false)

    return (ni=ni, xci=xci, xvi=xvi), thermal.T
end
