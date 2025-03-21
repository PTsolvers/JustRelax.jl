using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend_JR = CPUBackend

using JustPIC, JustPIC._2D

const backend = JustPIC.CPUBackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases)
        @index phases[1, i, j] = 1.0

        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex)
end



# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = Inf

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30            # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0e22           # viscosity
    G0 = 1.0e10         # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0e-14           # background strain-rate
    η_reg = 8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)

    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C,
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 9.81),
            CompositeRheology = CompositeRheology((visc, el_bg, )),
            Elasticity = el_bg,

        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    radius = 0.1
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(phase_ratios)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, CFL = 0.75 / √2.1)

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=@zeros(ni...), P=stokes.P))
    stokes.P        .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C =  @zeros(ni...))

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = DisplacementBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    stokes.U.Ux .= PTArray(backend_JR)([ x * εbg * lx * dt for x in xvi[1], _ in 1:(ny + 2)])
    stokes.U.Uy .= PTArray(backend_JR)([-y * εbg * ly * dt for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    displacement2velocity!(stokes, dt)
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 3.5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]
    strain_increment = true
    while it < 14


        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            strain_increment,
            igg;
            kwargs = (
                verbose          = false,
                iterMax          = 50e3,
                nout             = 1e2,
                viscosity_cutoff = (-Inf, Inf)
            )
        )
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

        x1 = 1e3.*(1:length(iters.norm_Rx)) 
        fig   = Figure(size = (1600, 1600), title = "t = $t , Strain Increment Approach")
        ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}", titlesize=35)
        # ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
        ax2   = Axis(fig[2,1], aspect = 1, title = L"E_{II}", titlesize=35)
        ax3   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize=35)
        ax4   = Axis(fig[2,2], aspect = 1)
        ax5 = Axis(fig[1, 3], title="Log10 Rx", xlabel="Iteration", ylabel="Error")
        ax6 = Axis(fig[2, 3], title="Log10 Ry", xlabel="Iteration", ylabel="Error")
        heatmap!(ax1, xci..., Array(stokes.τ.xx) , colormap=:batlow)
        # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
        # heatmap!(ax2, xci..., Array(log10.(stokes.EII_pl)) , colormap=:batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.xx)) , colormap=:batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        plot!(ax5, x1, Array(log10.(iters.norm_Rx)), color=:blue)
        plot!(ax6, x1, Array(log10.(iters.norm_Ry)), color=:blue)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end


n      = 256
nx     = n
ny     = n
figdir = "output/elastic_buildup_strainrate"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end


main(igg; figdir = figdir, nx = nx, ny = ny);
