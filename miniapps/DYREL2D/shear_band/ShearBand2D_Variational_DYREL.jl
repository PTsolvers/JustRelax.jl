using GeoParams, CairoMakie
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# Variational DYREL shear-band benchmark. Mirrors the APT variational miniapp
# `benchmarks/stokes2D/shear_band/ShearBand2D_variational.jl` (same physics, same
# RockRatio `ϕ`), but drives the solve through the variational `solve_DYREL!` path.

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
        if GGU.inside(p, circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0

        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
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

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ_fric = 30        # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / 2        # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 1.0e-2         # regularisation "viscosity"
    dt = η0 / G0 / 6.0     # assumes Maxwell time of 6
    el_bg = ConstantElasticity(; G = G0, Kb = 5)
    el_inc = ConstantElasticity(; G = Gi, Kb = 5)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ_fric),
        ϕ = ϕ_fric,
        η_vp = η_reg,
        Ψ = 0
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @rand(ni...)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    origin = 0.5, 0.5
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)
    air_phase = 0
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf); air_phase = air_phase
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # Time loop
    t, it = 0.0, 0
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]

    for _ in 1:20

        # Variational DYREL Stokes solver (ϕ positional, after phase_ratios) --
        iters = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                air_phase = air_phase,
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 10,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1,
                linear_viscosity = true,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = "convergence", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        lines!(ax2, iters.err_evo_it / nx, log10.(iters.err_evo_V), linewidth = 3, label = "V")
        lines!(ax2, iters.err_evo_it / nx, log10.(iters.err_evo_P), linewidth = 3, label = "P")
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        axislegend(ax2)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)

    end

    return nothing
end

n = 256
nx = n
ny = n
figdir = "Variational_ShearBands2D_DYREL"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
