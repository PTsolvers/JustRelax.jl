using GeoParams, GLMakie, CellArrays
using JustRelax
using JustRelax.JustRelax2D_AD
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)
using Enzyme

const backend = CPUBackend

using JustPIC, JustPIC._2D
const backend_JP = JustPIC.CPUBackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x-o_x)^2 + (y-o_y)^2) > radius^2
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        else
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius)
    @parallel (@idx ni.+1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius)
    return nothing
end

function init_phasesFD!(phase_ratios, xci, xvi, radius, FDxmin, FDxmax, FDymin, FDymax,di)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius, FDxmin, FDxmax, FDymin, FDymax,di)
        x, y = xc[i], yc[j]

            
        if (x <= FDxmax) && (x >= FDxmin) && (y <= FDymax) && (y >= FDymin) && ((x-o_x)^2 ≤ radius^2) && ((y-o_y)^2 ≤ radius^2)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 1.0
        elseif (x <= FDxmax) && (x >= FDxmin) && (y <= FDymax) && (y >= FDymin) 
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 1.0
            @index phases[4, i, j] = 0.0   
        elseif ((x-o_x)^2 ≤ radius^2) && ((y-o_y)^2 ≤ radius^2) #((x-o_x)^2 + (y-o_y)^2) ≤ radius^2
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0    # inclusion
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 0.0  
        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 0.0
            @index phases[4, i, j] = 0.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius, FDxmin, FDxmax, FDymin, FDymax,di)
    @parallel (@idx ni.+1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius, FDxmin, FDxmax, FDymin, FDymax,di)
    return nothing
end

using KahanSummation
# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs")

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
    G0      = 1.0           # elastic shear modulus
    Gi      = G0/(6.0-4.0)  # elastic shear modulus perturbation
    εbg     = 1.0           # background strain-rate
    η_reg   = 1e-2#8e-3          # regularisation "viscosity"
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg   = ConstantElasticity(; G=G0, Kb=4.0)
    el_inc  = ConstantElasticity(; G=Gi, Kb=4.0)
    visc    = LinearViscous(; η=η0)
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ,
        η_vp = η_reg,
        Ψ    = 0
    )

    visc    = LinearViscous(; η=η0)
    visc2    = LinearViscous(; η=2.0)
    visc3    = LinearViscous(; η=3.0)
    visc4    = LinearViscous(; η=4.0)

    el_bg_dG   = ConstantElasticity(; G=G0+0.001, Kb=4.0)
    el_inc_dG  = ConstantElasticity(; G=Gi+0.001, Kb=4.0)
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
        ),
        # Low density phase
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg_dG, pl)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc_dG, pl)),
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    radius       = 2*di[1]
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.95 / √2.1)

    # Adjoint 
    stokesAD = StokesArraysAdjoint(backend, ni)

    #indx     = findall((xci[1] .>= 0.5-di[1]) .& (xci[1] .<= 0.5+di[1]))
    #indy     = findall((xvi[2] .>= 0.7-di[2]) .& (xvi[2] .<= 0.7+di[2])) 
    indx     = findall((xci[1] .>= 4*di[1]) .& (xci[1] .<= 1.0-2*di[1]))
    indy     = findall((xvi[2] .>= 4*di[2]) .& (xvi[2] .<= 1.0-2*di[2])) 
    #indx     = findall((xci[1] .>= 0.48) .& (xci[1] .<= 0.52))
    #indy     = findall((xvi[2] .>= 0.7) .& (xvi[2] .<= 0.74))

    SensInd  = [indx, indy,]
    SensType = "Vy"

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)
    ρref = ρg[2]/1.0

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
    cost  = @zeros(length(xci[1]),length(xci[2]))  # cost function
    dp    = @zeros(length(xci[1]),length(xci[2]))  # parameter variation
    refcost = 0.0
    test    = 0.0
    param = 0.0


    ##########################
    ####### Preparing ########
    ##########################

    # while t < tmax
    for _ in 1:2

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
                ADout=plottingInt
            )
        );
        tensor_invariant!(stokes.ε)
        push!(τII, maximum(stokes.τ.xx))

        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

    end

    ##############################
    #### Reference Simulation ####
    ##############################

    stokesP       = deepcopy(stokes)
    ρgP           = deepcopy(ρg)
    phase_ratiosP = deepcopy(phase_ratios)
    rheologyP     = deepcopy(rheology)
    # Stokes solver ----------------
    Adjoint = adjoint_solve!(
        stokesP,
        stokesAD,
        pt_stokes,
        di,
        flow_bcs,
        ρg,
        phase_ratiosP,
        rheologyP,
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
    tensor_invariant!(stokes.ε)
    push!(τII, maximum(stokesP.τ.xx))
    
    t  += dt
    push!(sol, solution(εbg, t, G0, η0))
    push!(ttot, t)
        

    refcost = sum_kbn(BigFloat.(stokes.V.Vy[indx,indy]))

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
            radius       = 2*di[1]
            phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
            init_phasesFD!(phase_ratios, xci, xvi, radius, i, i+di[1], j, j+di[2],di)
 
            stokesP       = deepcopy(stokes)
            ρgP           = deepcopy(ρg)
            phase_ratiosP = deepcopy(phase_ratios)
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

        cost[xit,yit]  = sum_kbn(BigFloat.(stokesP.V.Vy[indx,indy]))
            
    
        println("it = $it; t = $t \n")

        it += 1

        (; η_vep, η) = stokesP.viscosity

        # Plotting ---------------------
        if it == 1 || rem(it, 40) == 0
            # visualisation
            th    = 0:pi/50:3*pi;
            xunit = @. radius * cos(th) + 0.5;
            yunit = @. radius * sin(th) + 0.5;

            fig   = Figure(size = (1600, 1600), title = "t = $t")
            ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}", titlesize=35)
            # ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
            ax2   = Axis(fig[2,1], aspect = 1, title = L"P", titlesize=35)
            ax3   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize=35)
            ax4   = Axis(fig[2,2], aspect = 1)
            ax5   = Axis(fig[3,1], aspect = 1, title = L"Sens fr", titlesize=35)
            ax6   = Axis(fig[3,2], aspect = 1, title = L"Sens G", titlesize=35)
            h1 = heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
            # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
            # heatmap!(ax2, xci..., Array(log10.(stokes.EII_pl)) , colormap=:batlow)
            h2 = heatmap!(ax2, xci..., Array(η) , colormap=:batlow)
            h3 = heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)) , colormap=:batlow)
            hfr = heatmap!(ax5, xci..., Array(stokesAD.fr) , colormap=:batlow)
            hG  = heatmap!(ax6, xci..., Array(stokesAD.G) , colormap=:batlow)
            #hfr = heatmap!(ax5, xci..., Array(stokesAD.VA.Vx) , colormap=:batlow)
            #hG  = heatmap!(ax6, xci..., Array(stokesAD.PA) , colormap=:batlow)
            lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
            lines!(ax4, ttot, τII, color = :black)
            lines!(ax4, ttot, sol, color = :red)
            Colorbar(fig[1,1][1,2], h1)
            Colorbar(fig[2,1][1,2], h2)
            Colorbar(fig[1,2][1,2], h3)
            Colorbar(fig[3,1][1,2], hfr)
            Colorbar(fig[3,2][1,2], hG)
            hidexdecorations!(ax1)
            hidexdecorations!(ax3)
            save(joinpath(figdir, "$(it).png"), fig)
        end

        end 

        
    end

    return refcost, cost, dp, Adjoint, ηref, ρref, stokesAD
end

n      = 32
nx     = n
ny     = n
figdir = "ShearBands2D_FD"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
refcost, cost, dp, Adjoint, ηref, ρref, stokesAD = main(igg; figdir = figdir, nx = nx, ny = ny);

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function plot_FD_vs_AD(refcost,cost,AD,nx,ny,ηref,ρref,stokesAD)

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

    xc   =  0.0  # x origin of block
    yc   =  0.0  # y origin of block
    r    =  2*di[1]  # half-width of block
    
    ind_block = findall(((Xc .-xc).^2 .≤ r^2) .& (((.-Yc) .- yc).^2 .≤ r.^2))
    sol_FD = @zeros(nx,ny)
    sol_FD .= (cost .- refcost) ./0.001
    #sol_FD[ind_block] = (cost[ind_block] .- refcost) ./ 0.015

    #sol_FD .= sol_FD .* ηref ./refcost
    #sol_FD .= sol_FD .* ρref  ./refcost

    ar = 1.0
    fig = Figure(size = (600, 600), title = "Compare Adjoint Sensitivities with Finite Difference Sensitivities")
    ax1 = Axis(fig[1,1], aspect = ar, title = "FD solution")
    ax2 = Axis(fig[2,1], aspect = ar, title = "Adjoint Solution")
    ax3 = Axis(fig[3,1], aspect = ar, title = "log10.(Error)")
    h1  = heatmap!(ax1, xci[1], xci[2], Array(sol_FD))
    h2  = heatmap!(ax2, xci[1], xci[2], Array(stokesAD.G) , colormap=:batlow)
    h3  = heatmap!(ax3, xci[1], xci[2], log10.(abs.(Array(sol_FD) .- Array(stokesAD.G))))
    hidexdecorations!(ax1)
    hidexdecorations!(ax2)
    hidexdecorations!(ax3)
    Colorbar(fig[1,1][1,2], h1)
    Colorbar(fig[2,1][1,2], h2)
    Colorbar(fig[3,1][1,2], h3)
    linkaxes!(ax1, ax2, ax3)    
    save("Comparison.png", fig)

    return sol_FD
end

FD = plot_FD_vs_AD(refcost,cost,AD,nx,ny,ηref,ρref,stokesAD)