include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint_variational/helper_functionsVS.jl")
include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint_variational/helperfunctionDis.jl")


sigmoid(x, k::Real) = 1 / (1 + exp(-k * x))

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs", f, run_param, run_ref, dp, dM,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot,Pdot,Tdot)

    # Physical domain ------------------------------------
    ly           = 1.0          # domain length in y
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
    Gi      = 0.5*G0           # shear modulus
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    Kb      = 3.0
    Kbb = 1.5

    visc_bg    = LinearViscous(; η=η0)
    visc_block = LinearViscous(; η=η0)
    el         = ConstantElasticity(G=G0, Kb = Kb)
    el_block   = ConstantElasticity(G=Gi, Kb = Kbb)

    # viscous and elastic blocks for parameter pertubation
    visc_bg_p    = LinearViscous(; η=η0)
    visc_block_p = LinearViscous(; η=η0)
    el_p         = ConstantElasticity(G=G0+dp, ν = 0.5)
    el_block_p   = ConstantElasticity(G=Gi+dp, ν = 0.5)

    MatrixM = DislocationCreep(A = 1e-2, n = 3.0, E = 1.0, V = 1e-2, r = 1.0, R = 1.0)
    BlockM  = DislocationCreep(A = 1e-1, n = 3.0, E = 1.0, V = 1e-2, r = 1.0, R = 1.0)

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_bg,el)),
            CompositeRheology = CompositeRheology((MatrixM,el)),            
            #CompositeRheology = CompositeRheology((visc_bg,el,pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_block,el)),
            CompositeRheology = CompositeRheology((BlockM,el_block)),
            #CompositeRheology = CompositeRheology((visc_block,el_block,pl)),
        ),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 1.0),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_bg_p,)),
            CompositeRheology = CompositeRheology((visc_bg_p,el_p)),
            #CompositeRheology = CompositeRheology((visc_bg_p,el_p,pl_p)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 1.5),
            Gravity           = ConstantGravity(; g = gr),
            #CompositeRheology = CompositeRheology((visc_block_p,)),
            CompositeRheology = CompositeRheology((visc_block_p,el_block_p)),
            #CompositeRheology = CompositeRheology((visc_block_p,el_block_p,pl_p)),
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
    ana     = false
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    #pt_stokes   = PTStokesCoeffs(li, di; ϵ_rel=1e-8, ϵ_abs=1e-8,  CFL = 0.95 / #√2.1)
    #pt_stokesAD = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-8, ϵ_rel = 1.0e-8, Re = 4.0π, r = 0.7, CFL = 0.98 / √2.1/1.0)

    pt_stokes   = PTStokesCoeffs(li, di; ϵ_rel = 1e-13, ϵ_abs=1e-14, Re = 4π, CFL = 0.95 / √2.1)
    #pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel = 1e-15, ϵ_abs=1e-15, Re = 1.0, r = 0.7, CFL = 0.95 / √2.1)
    #pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel = 1e-15, ϵ_abs=1e-15, Re = 2π, r = 0.7, CFL =  0.51 / √2.1)
    pt_stokesAD = PTStokesCoeffs(li, di; ϵ_rel = 1e-11, ϵ_abs=1e-14, Re = 4π, r = 0.7, CFL =  0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)
    #indx     = findall((xci[1] .>= 0.5-radius) .& (xci[1] .<= 0.5+radius)) .+ 1
    indx     = findall((xci[1] .>= 0.5-radius) .& (xci[1] .<= 0.5+radius)) .+ 1
    #indy     = findall((xvi[2] .>= 0.7) .& (xvi[2] .<= 0.8))  
    indy     = findall((xvi[2] .>= 0.5-radius) .& (xvi[2] .<= 0.5+radius))  
    ind      = vec([CartesianIndex(i, j) for i in indx, j in indy])
    SensInd  = ind
    SensType = "Vy"

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @ones(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

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

    stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2].-εbg/2.0)
    stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]].+εbg/2.0)

    #stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    #stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])

#=
    Vxtemp = PTArray(backend)([ 0.0 for x in xvi[1], _ in 1:ny+2])
    Vxtemp[1,:]   .= εbg/2.0 
    Vxtemp[end,:] .= -εbg/2.0
    #Vxtemp[2:end-1,:] .= 0.0
    Vytemp = PTArray(backend)([ 0.0 for _ in 1:nx+2, y in xvi[2]])
    Vytemp[:,1]   .= εbg/2.0
    Vytemp[:,end] .= -εbg/2.0
    #Vytemp[:,2:end-1] .= 0.0
    stokes.V.Vx .= Vxtemp
    stokes.V.Vy .= Vytemp
=#
    #stokes.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2])
    #stokes.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # IO -------------------------------------------------
    take(figdir)
    nout =1e2
    ##########################
    ####### Preparing ########
    ##########################
    plottingInt = 1  # plotting interval
    t, it      = 0.0, 0
    cost    = @zeros(length(xci[1]),length(xci[2])) # cost function
    refcost = 0.0
    
    # while t < tmax
    for _ in 1:4
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
                nout=nout,
                viscosity_cutoff = η_cutoff,
                verbose = false,
                ADout=1e8,
                accλ = accλ
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
    stokesRef.V.Vx .= PTArray(backend)([ x*εbg for x in xvi[1], _ in 1:ny+2].-εbg/2.0)
    stokesRef.V.Vy .= PTArray(backend)([-y*εbg for _ in 1:nx+2, y in xvi[2]].+εbg/2.0)

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
                nout=nout,
                viscosity_cutoff = η_cutoff,
                verbose = false,
                ADout=plottingInt,
                accλ = accλ,
                Vsc  = 1.0/2.0/1.0,
                Ptsc = 1.0/4.0/8.0,
                incom=true,
                βn = 0.0,
                pt_stokesAD = pt_stokesAD
            )
        );
        tensor_invariant!(stokesRef.ε_pl)
        if isCUDA
            CUDA.allowscalar() do
            refcost = sum_kbn(stokesRef.V.Vy[indx,indy])
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
        Kdot,
        igg;
        kwargs = (
            grid,
            origin,
            li,
            iterMax=150e3,
            nout=nout,
            viscosity_cutoff = η_cutoff,
            verbose = false,
            ADout=1e20,
            accλ = accλ,
            pt_stokesAD = pt_stokesAD,
            AdisDot,
            ndisDot,
            rdisDot,
            EdisDot,
            VdisDot,
            Pdot,
            Tdot   
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
                        cost[xit,yit]  = sum_kbn(stokesP.V.Vy[indx,indy]) 
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
        #cost = load(joinpath(figdir, "FD_solution.jld2"),"sol_FD_cpu")
    end
    return refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef, ρg, refcostdot,dt, stokes
end


#### Init Run ####
f         = 1    ; nx     = 16*f; ny     = 16*f
dp        = 1e-4
run_param = false
run_ref   = true
accλ      = true
visc    = true
dens    = false
Gdot    = false
frdot   = false
Kdot    = false
AdisDot = false
ndisDot = false
rdisDot = false
EdisDot = false
VdisDot = false
Pdot    = false
Tdot    = false
dM        = rand(Float64,nx,ny)
dM      ./= norm(dM)   # normalize M matrix
figdir    = "miniapps/adjoint_variational/Benchmark_Disl_Visc/"
#### Run ####
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
refcost, cost, dp, Adjoint, ηref, ρref, stokesAD, stokesRef, ρg, refcostdot,dt, stokes = main(igg; figdir = figdir, nx = nx, ny = ny,f,run_param, run_ref,dp, dM,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot,Pdot,Tdot); nothing
cost = rand(nx,ny)
plot_sens = stokesAD.G  #which sensitivity to plot
FD = plot_FD_vs_AD(refcost,cost,dp,plot_sens,nx,ny,ηref,ρref,stokesAD,figdir,f,Adjoint,stokesRef,run_param)


mach_ϵ = eps()

visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 1.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_K = hockeystickDis(refcost,stokesAD.K,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)

visc    = 0.0
dens    = 0.0
Gdot    = 1.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_G = hockeystickDis(refcost,stokesAD.G,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)


visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 1.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_ndis = hockeystickDis(refcost,stokesAD.ndis,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)


visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 1.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_Adis = hockeystickDis(refcost,stokesAD.Adis,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)

visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 1.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_rdis = hockeystickDis(refcost,stokesAD.rdis,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)

visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 1.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_Edis = hockeystickDis(refcost,stokesAD.Edis,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)

visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 1.0
Pdot    = 0.0
Tdot    = 0.0
epsilon, error_Vdis = hockeystickDis(refcost,stokesAD.Vdis,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)
include("/home/chris/Documents/2024_projects/JustRelax.jl/miniapps/adjoint_variational/PlottingPaper.jl")

visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 1.0
Tdot    = 0.0
epsilon, error_Pdis = hockeystickDis(refcost,stokesAD.Pcreep,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)


visc    = 0.0
dens    = 0.0
Gdot    = 0.0
frdot   = 0.0
Kdot    = 0.0
AdisDot = 0.0
ndisDot = 0.0
rdisDot = 0.0
EdisDot = 0.0
VdisDot = 0.0
Pdot    = 0.0
Tdot    = 1.0
epsilon, error_Tcreep = hockeystickDis(refcost,stokesAD.Tcreep,dp,dM,16,mach_ϵ,accλ,visc, dens, Gdot, frdot, Kdot, AdisDot, ndisDot, rdisDot, EdisDot, VdisDot, Pdot, Tdot, figdir)

#error_K = error_G
#error_G = error_η
#error_η = error_G


stokesAD.η .= rand(1)
error_η = error_G
make_final_plotDis(refcost,cost,dp,nx,ny,stokesAD,figdir,stokesRef)


plot_hockey(epsilon,error_G,error_K,error_Adis,error_Vdis,error_ndis,error_rdis, error_Edis,error_Pdis,error_Tcreep,figdir) 