using GeoParams, CairoMakie, CellArrays
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
        if ((x-o_x)^2 ≤ radius^2) && ((y-o_y)^2 ≤ radius^2)
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
    G0      = 1.0           # elastic shear modulus
    Gi      = G0/(6.0-4.0)+0.2  # elastic shear modulus perturbation
    εbg     = 1.0           # background strain-rate
    η_reg   = 8e-3          # regularisation "viscosity"
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg   = ConstantElasticity(; G=G0, Kb=4.0)
    el_inc  = ConstantElasticity(; G=Gi, Kb=4.0)
    #el_bg   = ConstantElasticity(; G=G0, ν=0.5)
    #el_inc  = ConstantElasticity(; G=Gi, ν=0.5)
    visc    = LinearViscous(; η=η0)
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ,
        η_vp = η_reg,
        Ψ    = 0
    )

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
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    radius       = 1*di[1]*f
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, radius)

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

    # while t < tmax
    for _ in 1:15



        # Stokes solver ----------------
        test = adjoint_solve!(
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

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

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
        #h2 = heatmap!(ax2, xci..., Array(stokes.P) , colormap=:batlow)
        h2 = heatmap!(ax2, xci..., Array(stokesAD.VA.Vx) , colormap=:batlow)
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

    return nothing
end

f = 2
n      = 16*f
nx     = n
ny     = n
figdir = "ShearBands2D"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny,f);