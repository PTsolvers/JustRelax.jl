## INCLUDE SCRIPTS FOR THE MODEL SETUP

include(joinpath(@__DIR__, "slab_setup.jl"))

## HELPER FUNCTIONS

if model.device == :cpu
    @eval(PTtype(::Type{Array}) = Array)
else
    @eval(PTtype(::Type{CuArray}) = CuArray)
end

@parallel_indices (i, j, k) function property2grid!(F, phase, Fi)
    F[i, j, k] = Fi[phase[i, j, k]]
    return nothing
end

function property2grid(phase, Fi, ; b_width=(1, 1, 1), smooth_iters=10)
    ni = size(phase)
    F = @zeros(ni...)

    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) property2grid!(F, phase, Fi)

    # smooth field
    F2 = deepcopy(F)
    if smooth_iters > 1
        for smooth_iters in 1:smooth_iters
            @hide_communication b_width begin # communication/computation overlap
                @parallel smooth!(F2, F, 1.0)
                F, F2 = F2, F
                update_halo!(F)
            end
        end
    end

    return F
end

## MAIN FUNCTION 

function DiffusionSlab(;
    ttot=5e6 * 3600 * 24 * 365,
    dt=500e3 * 3600 * 24 * 365,
    nx::Integer=32 - 1,
    ny::Integer=32 - 1,
    nz::Integer=32 - 1,
    lx=1000e3,
    ly=1000e3,
    lz=1000e3,
    b_width::NTuple{3,Integer}=(1, 1, 1),
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

    # Material phase and Temperature at cell centers
    phasec, Tc = generate_phases((xci ./ 1e3)...)
    phasev, = generate_phases((xvi ./ 1e3)...)

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)
    @parallel assign!(thermal.T, Tc)
    # general numerical coeffs for PT solver
    pt_thermal = PTThermalCoeffs(di, li; Resc=6π, CFL=0.8 / √3)

    ## Density
    ρm = 2700 # mantle
    ρl = 3300 # lithosphere
    ρmats = (ρm, ρl)
    ρ = property2grid(phasec, ρmats; b_width=b_width, smooth_iters=0)

    ## Temperature
    κmats = (8e-7, 8e-7)
    κ = property2grid(phasev, κmats; b_width=b_width, smooth_iters=0)
    Cp = (1200, 1200)
    _ρCpmats = @. 1 / Cp / ρmats
    _ρCp = property2grid(phasev, _ρCpmats; b_width=b_width, smooth_iters=0)
    thermal_parameters = ThermalParameters(κ, _ρCp)

    ## Boundary conditions
    thermal_bc = (flux_x=false, flux_y=true, flux_z=false)

    ## Time loop
    t = 0.0
    println("Starting solver")

    while t < ttot
        tic()
        solve!(thermal, pt_thermal, thermal_parameters, thermal_bc, ni, di, igg, dt)
        t_toc = toc()
        println("Done in $(t_toc) s")

        t += dt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi), thermal.T
end
