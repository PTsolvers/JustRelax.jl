include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint_variational/helper_functionsVS.jl")

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs", f, run_param, run_ref, dp, dM)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    εbg     = 1.0           # background strain-rate
    gr      = 0.0
    η0      = 1.0           # viscosity
    G0      = 1.0           # shear modulus
    Gi      = 0.5           # shear modulus
    ν0      = 0.45          # Poisson ratio
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    ana     = false

        # Physical properties using GeoParams ----------------
        τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
        ϕ = 30            # friction angle
        C = τ_y           # Cohesion
        η0 = 1.0           # viscosity
        G0 = 1.0           # elastic shear modulus
        Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
        εbg = 1.0           # background strain-rate
        η_reg = 8.0e-3          # regularisation "viscosity"
        dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    # viscous and elastic blocks for reference solve
    visc_bg    = LinearViscous(; η=1.0)
    visc_block = LinearViscous(; η=1.0)
    #el         = ConstantElasticity(G=G0, ν=ν0)
    #el_block   = ConstantElasticity(G=Gi, ν=ν0)
    el         = ConstantElasticity(G=G0, Kb = 4.0)
    el_block   = ConstantElasticity(G=Gi, Kb = 4.0)

    # viscous and elastic blocks for parameter pertubation
    visc_bg_p    = LinearViscous(; η=1.0+dp)
    visc_block_p = LinearViscous(; η=1.0+dp)
    #el_p         = ConstantElasticity(G=G0, ν=ν0)
    #el_block_p   = ConstantElasticity(G=Gi, ν=ν0)
    el_p         = ConstantElasticity(G=G0, Kb = 4.0)
    el_block_p   = ConstantElasticity(G=Gi, Kb = 4.0)

    # plascticity parameters
#    ϕ       = 30            # friction angle
#    C       = 1.6           # Cohesion
#    η_reg   = 8e-3 #1e-2          # regularisation "viscosity"
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
    C    = C / cosd(ϕ),
    ϕ    = ϕ,
    η_vp = η_reg,
    Ψ    = 15
    )

    pl_p      = DruckerPrager_regularised(;  # non-regularized plasticity
    C    = C / cosd(ϕ),
    ϕ    = ϕ,
    η_vp = η_reg,
    Ψ    = 15
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_bg,)),
            #CompositeRheology = CompositeRheology((visc_bg,el)),            
            CompositeRheology = CompositeRheology((visc_bg,el,pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_block,)),
            #CompositeRheology = CompositeRheology((visc_block,el_block)),
            CompositeRheology = CompositeRheology((visc_block,el_block,pl)),
        ),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_bg_p,)),
            #CompositeRheology = CompositeRheology((visc_bg_p,el_p)),
            CompositeRheology = CompositeRheology((visc_bg_p,el_p,pl_p)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_block_p,)),
            #CompositeRheology = CompositeRheology((visc_block_p,el_block_p)),
            CompositeRheology = CompositeRheology((visc_block_p,el_block_p,pl_p)),
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

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
    pt_stokes   = PTStokesCoeffs(li, di; ϵ_rel=1e-4, ϵ_abs=1e-12,  CFL = 0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)
    indx     = findall((xci[1] .>= 0.5-radius) .& (xci[1] .<= 0.5+radius)) .+ 1
    indy     = findall((xvi[2] .>= 0.5-1e-6) .& (xvi[2] .<= 0.5+1e-6))  
    ind      = vec([CartesianIndex(i, j) for i in indx, j in indy])
    SensInd  = ind
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

    Vxtemp = PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    Vxtemp[1:end-1,:] .= 0.0
    Vytemp = PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    Vytemp[:,1:end-1] .= 0.0
    stokes.V.Vx .= Vxtemp
    stokes.V.Vy .= Vytemp

    #stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    #stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # IO -------------------------------------------------
    take(figdir)

    ##########################
    ####### Preparing ########
    ##########################
    plottingInt = 1  # plotting interval
    t, it      = 0.0, 0
    cost    = @zeros(length(xci[1]),length(xci[2])) # cost function
    refcost = 0.0
    
    # while t < tmax
    for _ in 1:12
        # Stokes solver ----------------
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
            ana,
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
        t  += dt
        #plot_forward_solve(stokes,xci,ρg,t)
    end
    ρref     = deepcopy(ρg[2])./1.0
    
    ##############################
    #### Reference Simulation ####
    ##############################
    Adjoint = 1
    stokesRef     = deepcopy(stokes)
    ρgP           = deepcopy(ρg)
    phase_ratiosP = deepcopy(phase_ratios)
    stokesRef.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokesRef.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    if run_ref 
        # Stokes solver ----------------
        Adjoint = adjoint_solve_VariationalStokes!(
            stokesRef,
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
            ana,
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
        if isCUDA
            CUDA.allowscalar() do
            refcost = sum(stokesRef.V.Vy[indx,indy])
            end
        else
            refcost = Float64(sum_kbn((stokesRef.V.Vy[indx,indy])))
            #refcost = sum(stokesRef.V.Vy[indx,indy])
        end
    end
    tensor_invariant!(stokesRef.ε)
    (; η_vep, η) = stokes.viscosity
    ηref = η
    ρref     = deepcopy(ρg[2])./1.0

    #################################
    #### Dot product pertubation ####
    #################################
    stokesDot       = deepcopy(stokes)
    ρgDot           = deepcopy(ρg)
    phase_ratiosDot = deepcopy(phase_ratios)
    visc  = true
    dens  = false
    Gdot  = false
    frdot = false
    #stokesDot.viscosity.η .= stokesDot.viscosity.η + dM*dp
    # Stokes solver ----------------
    Dot = adjoint_solve_VariationalStokesDot!(
        stokesDot,
        stokesAD,
        pt_stokes,
        di,
        flow_bcs,
        ρgDot,
        phase_ratiosDot,
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
        dM,
        dp,
        visc,
        dens,
        Gdot,
        frdot,
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
    if isCUDA
        CUDA.allowscalar() do
        refcostdot = Float64(sum_kbn(stokesDot.V.Vy[indx,indy]))
        end
    else
        refcostdot = Float64(sum_kbn((stokesDot.V.Vy[indx, indy])))
    end
    
    #refcostdot=1.0

    ##########################
    #### Parameter change ####
    ##########################
    if run_param
        for (xit,i) in enumerate(xvi[1][1:end-1])
            for (yit,j) in enumerate(xvi[2][1:end-1])

            # Initialize phase ratios -------------------------------
                radius       = 1*di[1]*f
                phase_ratiosP = PhaseRatios(backend_JP, length(rheology), ni)
                init_phasesFD!(phase_ratiosP, xci, xvi, radius, i, i+di[1], j, j+di[2],di)
                stokesP       = deepcopy(stokes)
                ρgP           = deepcopy(ρg)

                compute_viscosity!(
                stokesP, phase_ratiosP, args, rheology, (-Inf, Inf)
                )
                compute_ρg!(ρgP[2], phase_ratiosP, rheology, args)
                # Stokes solver ----------------
                Adjoint = adjoint_solve_VariationalStokes!(
                    stokesP,
                    stokesAD,
                    pt_stokes,
                    di,
                    flow_bcs,
                    ρgP,
                    phase_ratiosP,
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
                    ana,
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
                if isCUDA
                    CUDA.allowscalar() do
                        cost[xit,yit]  = sum(stokesP.V.Vy[indx,indy]) 
                    end
                else
                    cost[xit,yit]  = sum_kbn((stokesP.V.Vy[indx,indy]))
                end
                println("it = $it \n")
                it += 1

            # Plotting ---------------------
                if it == 1 || rem(it, 100) == 0
                    #plot_forward_solve(stokesP,xci,ρgP,it)
                end
            end  
        end
    cost_cpu = Array(cost)
    jldsave(joinpath(figdir, "FD_cost.jld2"),cost_cpu=cost_cpu)
    else
        cost = load(joinpath(figdir, "FD_solution.jld2"),"sol_FD_cpu")
    end
    return refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef, ρg, refcostdot,dt
end

#### Init Run ####
f         = 4      ; nx     = 16*f; ny     = 16*f
dp        = 1e-6
run_param = true
run_ref   = true
dM        = rand(Float64,nx,ny)
dM      ./= norm(dM)   # normalize M matrix
figdir    = "miniapps/adjoint_variational/Test_VEP_eta/"
#### Run ####
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef, ρg, refcostdot,dt = main(igg; figdir = figdir, nx = nx, ny = ny,f,run_param, run_ref,dp, dM);
#cost .= rand(nx,ny)
plot_sens = stokesAD.η  #which sensitivity to plot
FD = plot_FD_vs_AD(refcost,cost,dp,plot_sens,nx,ny,ηref,ρref,stokesAD,figdir,f,Adjoint,stokesRef,run_param)


errorAD, FDs = hockeystick(refcost,refcostdot,plot_sens,dp,dM,FD,20)   # plot convergence test # plot convergence test

