const isCUDA = false
@static if isCUDA
    using CUDA
end
using JustRelax, JustRelax.JustRelax2D_AD, JustRelax.DataIO, Enzyme
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
include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint/Benchmarks_FD/helper_functions.jl")

# MAIN SCRIPT 
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

    visc_bg    = LinearViscous(; η=1.0)
    visc_block = LinearViscous(; η=10.0)

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_bg,)),),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_block,)),),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_bg,)),),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_block,)),),
    )

    # Initialize phase ratios -------------------------------
    radius       = 1*di[1]*f
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phasesFD!(phase_ratios, xci, xvi, radius, 100.0, 100.0, 100.0, 100.0,di)

    # RockRatios
    air_phase = 20
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-14,  CFL = 0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)
    indx     = findall((xci[1] .>= 0.5-radius/4) .& (xci[1] .<= 0.5+radius/4)) .+ 1
    indy     = findall((xvi[2] .>= 0.5-1e-6) .& (xvi[2] .<= 0.5+1e-6)) 
    SensInd  = [indx, indy,]
    SensType = "Vy"

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)

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
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # IO -------------------------------------------------
    take(figdir)

    plottingInt = 1
    it          = 0
    ##############################
    #### Reference Simulation ####
    ##############################
    adjoint_solve_VariationalStokes!(
        stokes,
        stokesAD,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratios,
        ϕ,
        rheology,
        args,
        dt,
        it, #Glit
        SensInd,
        SensType,
        grid,
        origin,
        li,
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
    return stokes, stokesAD, xci, xvi, di
end

#### Start Run ####
f      = 32
nx     = 16*f
ny     = 16*f
run_param = false
figdir = "miniapps/adjoint_variational/ConvergenceTest/"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
stokes, stokesAD, xci, xvi, di = main(igg; figdir = figdir, nx = nx, ny = ny,f,run_param);

sAD = round(sum(abs.(stokesAD.η)), digits=8)
sVx = round(sum(abs.(stokes.V.Vx)), digits=8)
sVy = round(sum(abs.(stokes.V.Vy)), digits=8)
sP  = round(sum(abs.(stokes.P)), digits=8)
sτ  = round(sum(abs.(stokes.τ.II)), digits=8)
sε  = round(sum(abs.(stokes.ε.II)), digits=8)

fig = Figure(size = (1000, 1000),fontsize=28);
ax1 = Axis(fig[1,1][1,1], aspect = 1, title= "sum Vx")
ax2 = Axis(fig[1,2][1,1], aspect = 1, title = "sum Vy")
ax3 = Axis(fig[2,1][1,1], aspect = 1, title = "sum P")
ax4 = Axis(fig[2,2][1,1], aspect = 1, title = "sum λVx")
ax5 = Axis(fig[3,1][1,1], aspect = 1, title = "sum λVy")
ax6 = Axis(fig[3,2][1,1], aspect = 1, title = "sum η")

h1 = heatmap!(ax1, xvi[1], xci[2], Array(stokes.V.Vx) , colormap=:lipari)
h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy) , colormap=:lipari)
h3 = heatmap!(ax3, xci[1], xci[2], Array(stokesAD.PA) , colormap=:lipari)
h4 = heatmap!(ax4, xci[1], xci[2], Array(stokesAD.VA.Vx) , colormap=:lipari)
h5 = heatmap!(ax5, xci[1], xci[2], Array(stokesAD.VA.Vy), colormap=:lipari)
h6 = heatmap!(ax6, xci[1], xci[2], Array(stokesAD.η) , colormap=:lipari)

Colorbar(fig[1,1][1,2], h1, height=Relative(0.8))
Colorbar(fig[1,2][1,2], h2, height=Relative(0.8))
Colorbar(fig[2,1][1,2], h3, height=Relative(0.8))
Colorbar(fig[2,2][1,2], h4, height=Relative(0.8))
Colorbar(fig[3,1][1,2], h5, height=Relative(0.8))
Colorbar(fig[3,2][1,2], h6, height=Relative(0.8))
save(joinpath(figdir, "Convergence$nx.png"), fig)

