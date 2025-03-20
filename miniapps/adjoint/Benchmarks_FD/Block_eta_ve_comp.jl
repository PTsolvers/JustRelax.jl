using GeoParams, CairoMakie, CellArrays, JustRelax
using JustRelax.JustRelax2D_AD, ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
using Enzyme, KahanSummation
const backend = CPUBackend
using JustPIC, JustPIC._2D
const backend_JP = JustPIC.CPUBackend
include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint/Benchmarks_FD/helper_functions.jl")

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs",f)

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
    εbg     = 0.0           # background strain-rate
    G0      = 1.0           # elastic shear modulus
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    el      = ConstantElasticity(G=G0, ν=0.45)
    visc_bg    = LinearViscous(; η=1.0)
    visc_block = LinearViscous(; η=100.0)

    # parameter pertubation
    dp = 1e-4
    visc_bg_p   = LinearViscous(; η=1.0+dp)
    visc_block_p = LinearViscous(; η=100.0+dp)

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_bg,el)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_block,el)),
        ),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_bg_p,el)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((visc_block_p,el)),
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
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)
    #indx     = findall((xci[1] .>= 0.48) .& (xci[1] .<= 0.52))
    #indy     = findall((xvi[2] .>= 0.7) .& (xvi[2] .<= 0.74))
    indx     = findall((xci[1] .>= di[1]) .& (xci[1] .<= 1.0-(di[1])))
    indy     = findall((xvi[2] .>= di[2]) .& (xvi[2] .<= 1.0-(di[2]))) 
    SensInd  = [indx, indy,]
    SensType = "Vy"

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
    cost    = BigFloat.(@zeros(length(xci[1]),length(xci[2])))  # cost function
    refcost = 0.0
    test    = 0.0
    param   = 0.0

    ##########################
    ####### Preparing ########
    ##########################
    # while t < tmax
    for _ in 1:4

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
    refcost = sum_kbn(BigFloat.(stokesRef.V.Vy[indx.+1,indy]))

    (; η_vep, η) = stokes.viscosity
    ηref = η
    ##scale η sensitivity
    #AD.ηb .= AD.ηb .* ηref ./ refcost
    #AD.ρb .= AD.ρb .* ρref ./ refcost
    
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
            cost[xit,yit]  = sum_kbn(BigFloat.(stokesP.V.Vy[indx.+1,indy]))
            println("it = $it \n")
            it += 1

        (; η_vep, η) = stokesP.viscosity

        # Plotting ---------------------
        if it == 1 || rem(it, 40) == 0
            plot_forward_solve(stokesP,xci,ρgP,it)
        end
        end  
    end
    
    return refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef
end

#### Start Run ####
f      = 1
nx     = 16*f
ny     = 16*f
figdir = "miniapps/adjoint/Benchmarks_FD/Block_eta_ve"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef = main(igg; figdir = figdir, nx = nx, ny = ny,f);

#### Plotting Comparison ####
function plot_FD_vs_AD(refcost,cost,dp,AD,nx,ny,ηref,ρref,stokesAD,figdir,f, Adjoint, Ref)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cellsvertices of the cells
    Xc, Yc = meshgrid(xci[1], xci[2])

    #ind = findall(xci[2] .≤ 0.29)

    o_x   =  0.5  # x origin of block
    o_y   =  0.5  # y origin of block
    r    =  1*di[1]*f  # half-width of block
    
    ind_block = findall(((Xc.-o_x).^2 .≤ r^2) .& ((Yc.-o_y).^2 .≤ r.^2))
    sol_FD = @zeros(nx,ny)
    sol_FD .= ((cost .- refcost)./dp)#./(abs(refcost)) ./ (di[1] * di[2])
    #sol_FD[ind_block] .*= 0.5
    #sol_FD .= ((cost .- refcost)./dp) * di[1] * di[2] #./ refcost
    #sol_FD[ind_block] = (cost[ind_block] .- refcost) ./ 0.0005

    #AD = deepcopy(stokesAD.G)
    AD = Adjoint.ηb
    #AD_G .*= 2.0
    #AD_G[ind_block] ./= 2.0
    #AD_G[ind_block] .*= 0.5
    AD .= AD# ./ abs(refcost) #./(di[1] * di[2]) 

    #sol_FD .= sol_FD .* ηref ./refcost
    #sol_FD .= sol_FD .* ρref  ./refcost


    # scale the Sensitivities
    AD_norm = (AD .+ abs(minimum(AD)))
    AD_norm .= AD_norm ./ maximum(AD_norm)

    sol_FD_norm = (sol_FD .+ abs(minimum(sol_FD)))
    sol_FD_norm .= sol_FD_norm ./ maximum(sol_FD_norm)

    sumFD = round(sum(abs.(sol_FD)),digits=2)
    sumAD = round(sum(abs.(AD)),digits=2)
    fig = Figure(size = (640, 1000), title = "Compare Adjoint Sensitivities with Finite Difference Sensitivities",fontsize=16)
    ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}")
    ax2   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})")
    ax3   = Axis(fig[2,1], aspect = 1, title = L"Vx")
    ax4   = Axis(fig[2,2], aspect = 1, title = L"Vy")
    ax5   = Axis(fig[3,1], aspect = 1, title = L"λ Vx")
    ax6   = Axis(fig[3,2], aspect = 1, title = L"λ Vy")
    ax7 = Axis(fig[4,1], aspect = 1, title = "FD Sens. sum(abs)=$sumFD",titlesize=16)
    ax8 = Axis(fig[4,2], aspect = 1, title = "AD Sens. sum(abs)=$sumAD",titlesize=16)
    ax9 = Axis(fig[5,1], aspect = 1, title = "Error",titlesize=16)
    h1 = heatmap!(ax1, xci..., Array(Ref.τ.II) , colormap=:managua)
    h2 = heatmap!(ax2, xci..., Array(log10.(Ref.ε.II)) , colormap=:managua)
    #Vx_range = maximum(abs.(Ref.V.Vx))
    #Vy_range = maximum(abs.(Ref.V.Vy))
    h3  = heatmap!(ax3, xci[1], xci[2], Array(Ref.V.Vx),colormap=:roma)#,colorrange=(-Vx_range,Vx_range))
    h4  = heatmap!(ax4, xci[1], xci[2], Array(Ref.V.Vy),colormap=:roma)#,colorrange=(-Vy_range,Vy_range))
    #λVx_range = maximum(abs.(stokesAD.VA.Vx))
    #λVy_range = maximum(abs.(stokesAD.VA.Vy))
    h5  = heatmap!(ax5, xci[1], xci[2], Array(stokesAD.VA.Vx),colormap=:lipari)#, colorrange=(-λVx_range,λVx_range))
    h6  = heatmap!(ax6, xci[1], xci[2], Array(stokesAD.VA.Vy),colormap=:lipari)#, colorrange=(-λVy_range,λVy_range))
    h7  = heatmap!(ax7, xci[1], xci[2], Array(sol_FD),colormap=:lipari)
    h8  = heatmap!(ax8, xci[1], xci[2], Array(AD),colormap=:lipari)
    h9  = heatmap!(ax9, xci[1], xci[2], Array(sol_FD_norm .- AD_norm),colormap=:jet)
    hidexdecorations!(ax1);hidexdecorations!(ax2);hidexdecorations!(ax3);hidexdecorations!(ax4);hidexdecorations!(ax5)
    hidexdecorations!(ax6);hideydecorations!(ax2);hideydecorations!(ax4);hideydecorations!(ax6);hideydecorations!(ax8)

    Colorbar(fig[1,1][1,2], h1, height=Relative(0.8))
    Colorbar(fig[1,2][1,2], h2, height=Relative(0.8))
    Colorbar(fig[2,1][1,2], h3, height=Relative(0.8))
    Colorbar(fig[2,2][1,2], h4, height=Relative(0.8))
    Colorbar(fig[3,1][1,2], h5, height=Relative(0.8))
    Colorbar(fig[3,2][1,2], h6, height=Relative(0.8))
    Colorbar(fig[4,1][1,2], h7, height=Relative(0.8))
    Colorbar(fig[4,2][1,2], h8, height=Relative(0.8))
    Colorbar(fig[5,1][1,2], h9, height=Relative(0.8))
    colsize!(fig.layout, 1, Aspect(1, 1.4))
    colsize!(fig.layout, 2, Aspect(1, 1.4))
    colgap!(fig.layout, 8)
    #rowgap!(fig.layout, 4.0)
    #linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)    
    save(joinpath(figdir, "Comparison.png"), fig)

    return sol_FD
end

FD = plot_FD_vs_AD(refcost,cost,dp,AD,nx,ny,ηref,ρref,stokesAD,figdir,f,Adjoint,stokesRef)
