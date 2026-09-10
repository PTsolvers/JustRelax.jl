# Shear-band benchmark (visco-elasto-plastic inclusion in simple shear) solved with the DYREL
# Stokes solver, and run with the adjoint solve enabled.
#
# The forward problem is the same as `miniapps/DYREL2D/shear_band/ShearBand2D_DYREL.jl`:
# a weak circular inclusion in a Drucker-Prager matrix loaded by a background strain rate
# `εbg`, whose bulk stress is compared against the analytic visco-elastic loading curve
# `2 ε η (1 - exp(-G t / η))`. On top of that, `solve_DYREL!` is handed an
# `AdjointStokesArrays` and `adjoint = true`, so after the forward iterations converge it
# also runs the reverse pass for an observation region placed over the inclusion.

const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

# activate the miniapps environment *before* loading anything from it, and resolve the path
# relative to this file so the script does not depend on the working directory
using Pkg
# Pkg.activate(joinpath(@__DIR__, "..", "..", "..", "miniapps"))

using JustRelax, JustRelax.JustRelax2D

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

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using GeoParams, CairoMakie

import JustPIC.GridGeometryUtils as GGU

# HELPER FUNCTIONS ----------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the phase ratios
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
function main(
        igg;
        nx = 64,
        ny = 64,
        figdir = "ShearBands2D_adjoint",
        nt = 15,
        viscosity_perturbation = 0.0,
        η_multiplier = nothing,
        adjoint = true,
        return_fields = false,
        plot_results = true,
        solver_ϵ = 1.0e-6,
        verbose = true,
    )

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly             # domain length in x
    ni = nx, ny         # number of cells
    li = lx, ly         # domain length in x- and y-
    di = @. li / ni     # grid step in x- and -y
    origin = 0.0, 0.0       # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid           # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y = 1.6            # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30             # friction angle
    C = τ_y            # Cohesion
    η0_base = 1.0       # unperturbed viscosity
    η0 = η0_base * exp(viscosity_perturbation)
    G0 = 1.0            # elastic shear modulus
    Gi = G0 / 2         # elastic shear modulus perturbation
    εbg = 1.0            # background strain-rate
    η_reg = 1.0e-2         # regularisation "viscosity"
    dt = η0_base / G0 / 6.0  # keep the physical time step fixed during gradient tests
    el_bg = ConstantElasticity(; G = G0, Kb = 5)
    el_inc = ConstantElasticity(; G = Gi, Kb = 5)
    visc_bg = LinearViscous(; η = η0_base)
    visc_inc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_bg, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_inc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    inclusion = 0.5, 0.5
    circle = GGU.Circle(inclusion, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Adjoint Stokes --------------------------------------
    stokes_ad = AdjointStokesArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces -- this benchmark is driven purely by the boundary strain rate, so both
    # phases are weightless and ρg stays zero
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0e0
    @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0e0
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Measure Vy in the upper half of the inclusion. Summing over the complete, symmetric
    # inclusion makes this objective almost invariant to the symmetric viscosity direction
    # used by the Taylor test, leaving only solver noise in the finite difference.
    observation = (;
        field = :Vy,
        center = (inclusion[1], inclusion[2] + radius / 2),
        half_width = (radius, radius / 2),
    )

    # IO -------------------------------------------------
    plot_results && take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = solver_ϵ)

    # Time loop
    t, it = 0.0, 0
    τII = [0.0e0]
    sol = [0.0e0]
    ttot = [0.0e0]

    for it in 1:nt

        AdjointSolve = adjoint && it == nt
        step_η_multiplier = it == nt ? η_multiplier : nothing
        # Stokes solver ----------------
        iters = solve_DYREL!(
            stokes,
            stokes_ad,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            verbose_PH = verbose,
            verbose_DR = false,
            iterMax = 50.0e3,
            nout = 10,
            rel_drop = 1.0e-2,
            λ_relaxation_PH = 1,
            λ_relaxation_DR = 1,
            viscosity_relaxation = 1,
            linear_viscosity = true,
            η_multiplier = step_η_multiplier,
            viscosity_cutoff = (-Inf, Inf),
            adjoint = AdjointSolve,
            observation = observation,
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        t += dt

        push!(τII, maximum(stokes.τ.xx))
        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        verbose && println("it = $it; t = $t \n")

        if plot_results && it in (1, nt)
            # Plotting ---------------------
            # forward and adjoint velocities live on the same staggered grid, so both get
            # interpolated to the vertices the same way
            Vx_v = @zeros(ni .+ 1...)
            Vy_v = @zeros(ni .+ 1...)
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            λVx_v = @zeros(ni .+ 1...)
            λVy_v = @zeros(ni .+ 1...)
            velocity2vertex!(λVx_v, λVy_v, stokes_ad.λV.Vx, stokes_ad.λV.Vy)

            # a constant field (all-zero adjoint arrays, say) has no colour range Makie can use
            function crange(A, diverging)
                if diverging
                    m = maximum(abs, A)
                    return iszero(m) ? (-1.0, 1.0) : (-m, m)
                end
                lo, hi = extrema(A)
                return lo == hi ? (lo - 1, hi + 1) : (lo, hi)
            end

            f = Figure(size = (1900, 950))
            function panel!(row, col, title, x, A; colormap = :batlow, diverging = false)
                ax = Axis(
                    f[row, 2col - 1];
                    aspect = DataAspect(), title = title,
                    xlabel = row == 2 ? "x" : "", ylabel = col == 1 ? "y" : "",
                )
                h = heatmap!(ax, x..., Array(A); colormap = colormap, colorrange = crange(Array(A), diverging))
                Colorbar(f[row, 2col], h)
                row == 1 && hidexdecorations!(ax, grid = false)
                col > 1 && hideydecorations!(ax, grid = false)
                return nothing
            end

            # forward Stokes solution -- this benchmark is weightless (ρ ≡ 0), so the second
            # invariant of the stress takes the slot the density occupies in the sinking block
            panel!(1, 1, "τII", xci, stokes.τ.II; colormap = :batlow)
            panel!(1, 2, "Vx", xvi, Vx_v; colormap = :vik, diverging = true)
            panel!(1, 3, "Vy", xvi, Vy_v; colormap = :vik, diverging = true)
            panel!(1, 4, "P", xci, stokes.P; colormap = :batlow)

            # adjoint solution
            panel!(2, 1, "λVx", xvi, λVx_v; colormap = :vik, diverging = true)
            panel!(2, 2, "λVy", xvi, λVy_v; colormap = :vik, diverging = true)
            panel!(2, 3, "ρ (adjoint)", xci, stokes_ad.ρ; colormap = :roma)
            panel!(2, 4, "η (adjoint)", xci, stokes_ad.viscosity.η; colormap = :batlow)

            save(joinpath(figdir, "$(it).png"), f)

            # benchmark curve: bulk stress against the analytic visco-elastic loading solution
            fs = Figure(size = (1200, 500))
            axs1 = Axis(fs[1, 1], title = "stress build-up", xlabel = "t", ylabel = L"\tau_{xx}")
            lines!(axs1, ttot, τII, color = :black, linewidth = 3, label = "DYREL")
            lines!(axs1, ttot, sol, color = :red, linewidth = 3, linestyle = :dash, label = "analytic")
            axislegend(axs1; position = :rb)
            axs2 = Axis(fs[1, 2], title = "convergence", xlabel = "iterations / nx", ylabel = L"\log_{10}(err)")
            lines!(axs2, iters.err_evo_it / nx, log10.(iters.err_evo_V), linewidth = 3, label = "V")
            lines!(axs2, iters.err_evo_it / nx, log10.(iters.err_evo_P), linewidth = 3, label = "P")
            axislegend(axs2)
            save(joinpath(figdir, "stress_evolution_$(it).png"), fs)
        end
    end

    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    cost_field = observation.field === :Vx ? stokes.V.Vx : observation.field === :Vy ? stokes.V.Vy : stokes.P
    cost = sum(@view(cost_field[mask.i, mask.j]))
    phase2 = map(ratio -> ratio[2], Array(phase_ratios.center))
    phase2_vertex = map(ratio -> ratio[2], Array(phase_ratios.vertex))
    return return_fields ? (;
            cost,
            viscosity_gradient = Array(stokes_ad.viscosity.η),
            viscosity_gradient_vertex = Array(stokes_ad.viscosity.ηv),
            viscosity = Array(stokes.viscosity.η),
            viscosity_vertex = Array(stokes.viscosity.ηv),
            viscosity_direction = Array(stokes.viscosity.η) .* phase2,
            viscosity_direction_vertex = Array(stokes.viscosity.ηv) .* phase2_vertex,
        ) : nothing
end

# Define `NO_AUTORUN` before including this file to load the functions without running the
# demo (see ViscosityGradientTest.jl). A plain `include` from the REPL still runs it.
if !@isdefined(NO_AUTORUN)
    nx = 32
    ny = 32
    figdir = "ShearBands2D_adjoint"
    # A global grid may still be active from an earlier run in this session, which makes
    # `init_global_grid` throw. Tear it down first, keeping MPI alive so it can be re-created.
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    @time main(igg; figdir = figdir, nx = nx, ny = ny)
end
