const isCUDA = false

@static if isCUDA
    using CUDA
end
using JustRelax, JustRelax.JustRelax2D_AD, JustRelax.DataIO
using GeoParams, CairoMakie, CellArrays, JLD2
const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end
using ParallelStencil, ParallelStencil.FiniteDifferences2D
@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using JustPIC, JustPIC._2D
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

#using GeoParams, CairoMakie, CellArrays, JustRelax
#@init_parallel_stencil(Threads, Float64, 2)
using Enzyme, KahanSummation, AccurateArithmetic
#const backend = CPUBackend
#using JustPIC, JustPIC._2D
#const backend_JP = JustPIC.CPUBackend
include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint/Benchmarks_FD/helper_functions.jl")

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs",f,run_param)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt           = Inf

    # Physical properties using GeoParams ----------------
    #τ_y     = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30            # friction angle
    C       = 1.6           # Cohesion
    η0      = 1.0           # viscosity
    εbg     = 1.0           # background strain-rate
    G0      = 1.0           # elastic shear modulus
    Gi      = G0/(6.0-4.0)  # elastic shear modulus perturbation
    η_reg   = 8e-3 #1e-2          # regularisation "viscosity"
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    visc_bg    = LinearViscous(; η=1.0)
    visc_block = LinearViscous(; η=1.0)
    el         = ConstantElasticity(G=G0, Kb=4.0)
    el_inc     = ConstantElasticity(G=Gi, Kb=4.0)
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
    C    = C,
    ϕ    = ϕ,
    η_vp = η_reg,
    Ψ    = 0)

    # parameter pertubation
    dp = 1e-2
    visc_p    = LinearViscous(; η=1.0+dp)


    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_bg,el,pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_block,el_inc,pl)),
        ),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_p,el,pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_p,el_inc,pl)),
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    radius       = 1*di[1]*f
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phasesFD!(phase_ratios, xci, xvi, radius, 100.0, 100.0, 100.0, 100.0,di)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-14,  CFL = 0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)
    #indx     = findall((xci[1] .>= 0.48) .& (xci[1] .<= 0.52))
    #indy     = findall((xvi[2] .>= 0.7) .& (xvi[2] .<= 0.74))
    indx     = findall((xci[1] .>= di[1]) .& (xci[1] .<= 1.0-(di[1])))
    indy     = findall((xvi[2] .>= di[2]) .& (xvi[2] .<= 1.0-(di[2]))) 
    SensInd  = [indx, indy,]
    SensType = "Vy"
    #Xc, Yc = meshgrid(xci[1], xci[2])

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

    # Rheology
    η_cutoff = -Inf, Inf
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs     = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    plottingInt = 1  # plotting interval
    t, it      = 0.0, 0
    tmax       = 5
    τII        = Float64[]
    sol        = Float64[]
    ttot       = Float64[]

    # init matrixes for FD sensitivity test
    #cost    = BigFloat.(@zeros(length(xci[1]),length(xci[2])))  # cost function
    cost    = @zeros(length(xci[1]),length(xci[2])) # cost function
    refcost = 0.0
    test    = 0.0
    param   = 0.0

    ##########################
    ####### Preparing ########
    ##########################
    # while t < tmax
    for _ in 1:8

        # Stokes solver ----------------
        adjoint_solve!(
            stokes,
            stokesAD,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            it, #Glit
            SensInd,
            SensType,
            igg;
            kwargs = (
                grid,
                origin,
                li,
                iterMax=150e3,
                nout=1e3,
                viscosity_cutoff = η_cutoff,
                verbose = false,
                ADout=1e8
            )
        );
        tensor_invariant!(stokes.ε)
        t  += dt
        plot_forward_solve(stokes,xci,ρg,t)
    end

    ρref     = deepcopy(ρg[2])./1.0

    ##############################
    #### Reference Simulation ####
    ##############################
    stokesRef     = deepcopy(stokes)
    ρgP           = deepcopy(ρg)
    phase_ratiosP = deepcopy(phase_ratios)
    # Stokes solver ----------------
    Adjoint = adjoint_solve!(
        stokesRef,
        stokesAD,
        pt_stokes,
        di,
        flow_bcs,
        ρgP,
        phase_ratiosP,
        rheology,
        args,
        dt,
        it, #Glit
        SensInd,
        SensType,
        igg;
        kwargs = (
            grid,
            origin,
            li,
            iterMax=150e3,
            nout=1e3,
            viscosity_cutoff = η_cutoff,
            verbose = false,
            ADout=plottingInt
        )
    );
    tensor_invariant!(stokesRef.ε)
    #refcost = sum_kbn(BigFloat.(stokesRef.V.Vy[indx.+1,indy]))
    if isCUDA
        CUDA.allowscalar() do
            refcost = sum(stokesRef.V.Vy[indx.+1,indy])
        end
    else
        refcost = sum(stokesRef.V.Vy[indx.+1,indy])
    end

    (; η_vep, η) = stokes.viscosity
    ηref = η
    ##scale η sensitivity
    #AD.ηb .= AD.ηb .* ηref ./ refcost
    #AD.ρb .= AD.ρb .* ρref ./ refcost
    if (run_param)
    ##########################
    #### Parameter change ####
    ##########################
    for (xit,i) in enumerate(xvi[1][1:end-1])
        for (yit,j) in enumerate(xvi[2][1:end-1])

        # Initialize phase ratios -------------------------------
            radius       = 1*di[1]*f
            phase_ratiosP = PhaseRatios(backend_JP, length(rheology), ni)
            init_phasesFD!(phase_ratiosP, xci, xvi, radius, i, i+di[1], j, j+di[2],di)
 
            stokesP       = deepcopy(stokes)
            ρgP           = deepcopy(ρg)
            # Stokes solver ----------------
            param = adjoint_solve!(
                stokesP,
                stokesAD,
                pt_stokes,
                di,
                flow_bcs,
                ρgP,
                phase_ratiosP,
                rheology,
                args,
                dt,
                it, #Glit
                SensInd,
                SensType,
                igg;
                kwargs = (
                    grid,
                    origin,
                    li,
                    iterMax=150e3,
                    nout=1e3,
                    viscosity_cutoff = η_cutoff,
                    verbose = false,
                    ADout=1e20
                )
            );
            tensor_invariant!(stokesP.ε)
            #cost[xit,yit]  = sum_kbn(BigFloat.(stokesP.V.Vy[indx.+1,indy]))
            if isCUDA
                CUDA.allowscalar() do
                    cost[xit,yit]  = sum(stokesP.V.Vy[indx.+1,indy])
                end
            else
                cost[xit,yit]  = sum(stokesP.V.Vy[indx.+1,indy])
            end
            println("it = $it \n")
            it += 1

        (; η_vep, η) = stokesP.viscosity

        # Plotting ---------------------
        if it == 1 || rem(it, 100) == 0
            plot_forward_solve(stokesP,xci,ρgP,it)
        end
        end  
    end
    cost_cpu = Array(cost)
    jldsave(joinpath(figdir, "FD_cost.jld2"),cost_cpu=cost_cpu)
    else
        #cost = load(joinpath(figdir, "FD_solution.jld2"),"sol_FD_cpu")
        cost = load(joinpath(figdir, "FD_cost.jld2"),"cost_cpu")
    end
    return refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef
end

#### Start Run ####
f      = 1
nx     = 16*f
ny     = 16*f
run_param = true
figdir = "miniapps/adjoint/Benchmarks_FD/Shear_eta_vep_comp"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef = main(igg; figdir = figdir, nx = nx, ny = ny,f,run_param);

#which sensitivity to plot
plot_sens = stokesAD.η
FD = plot_FD_vs_AD(refcost,cost,dp,plot_sens,nx,ny,ηref,ρref,stokesAD,figdir,f,Adjoint,stokesRef)
