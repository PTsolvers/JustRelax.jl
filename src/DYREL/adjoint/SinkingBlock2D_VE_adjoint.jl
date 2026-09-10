const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
# using Pkg; Pkg.activate("miniapps")

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
using GeoParams
using CairoMakie:
    Axis,
    Colorbar,
    DataAspect,
    Figure,
    heatmap!,
    hidexdecorations!,
    hideydecorations!,
    save

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Thermal rectangular perturbation

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, xc, yc, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            # plume - rectangular
            @index phases[ip, i, j] = if ((x - xc)^2 ≤ r^2) && ((depth - yc)^2 ≤ r^2)
                2.0
            else
                1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, xc, yc, r)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_j(z))
    return nothing
end

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function sinking_block2D_VE(
        igg;
        ar = 8,
        ny = 16,
        nx = ny * 8,
        figdir = "SinkingBlock2D_VE_adjoint",
        thermal_perturbation = :circular,
        nt = 10,
        viscosity_perturbation = 0.0,
        η_multiplier = nothing,
        adjoint = true,
        return_fields = false,
        plot_results = true,
        solver_ϵ = 1.0e-6,
        verbose = true,
    )

    # Nondimensional domain ------------------------------
    ly = 1.0
    lx = ly * ar
    origin = -lx / 2, -ly                         # origin coordinates
    ni = nx, ny                           # number of cells
    li = lx, ly                           # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Nondimensional material properties -----------------
    ρ_mantle = 1.0
    ρ_block = 1.2
    η_mantle = 1.0
    η_block = 10.0
    gravity = 1.0
    G = 0.1
    # NOTE: the elastic element has to sit *inside* the `CompositeRheology` tuple. `get_G`
    # reads the composite, not the `Elasticity` field, so a `MaterialParams` that only sets
    # `Elasticity` yields G = 0 -> Inf, i.e. a purely viscous model.
    elasticity = ConstantElasticity(; G = G, Kb = 5G)
    rheology = (
        SetMaterialParams(;
            Name = "Mantle",
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ_mantle),
            CompositeRheology = CompositeRheology(
                (LinearViscous(; η = η_mantle * exp(viscosity_perturbation)), elasticity)
            ),
            Elasticity = elasticity,
            Gravity = ConstantGravity(; g = gravity),
        ),
        SetMaterialParams(;
            Name = "Block",
            Phase = 2,
            Density = ConstantDensity(; ρ = ρ_block),
            CompositeRheology = CompositeRheology(
                (LinearViscous(; η = η_block * exp(viscosity_perturbation)), elasticity)
            ),
            Elasticity = elasticity,
            Gravity = ConstantGravity(; g = gravity),
        ),
    )
    # One mantle Maxwell time keeps both viscous and elastic deformation active.
    dt = η_mantle / G
    # ----------------------------------------------------

    grid_vxi = velocity_grids(xci, xvi, di)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 40, 40, 12
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    # temperature
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    # Rectangular density anomaly
    xc_anomaly = 0.0      # centred horizontally (domain spans -lx/2 .. lx/2)
    yc_anomaly = -ly / 4  # centred vertically; `init_phases!` takes it as a depth
    r_anomaly_x = 0.2
    r_anomaly_y = 0.1
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    # location of the plot
    xc_anomaly_block = 0.0      # centred horizontally (domain spans -lx/2 .. lx/2)
    yc_anomaly_block = -ly / 2  # centred vertically; `init_phases!` takes it as a depth
    r_anomaly_block = 0.1
    init_phases!(pPhases, particles, xc_anomaly_block, abs(yc_anomaly_block), r_anomaly_block)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Adjoint Stokes --------------------------------------
    stokes_ad = AdjointStokesArrays(backend, ni)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = @ones(ni .+ 2...), P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt = dt)
    viscosity_cutoff = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    # ----------------------------------------------------

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    plot_results && take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = solver_ϵ)
    observation = (;
        field = :Vy,
        center = (xc_anomaly, yc_anomaly),
        half_width = (r_anomaly_x, r_anomaly_y),
    )

    it = 0 # iteration counter
    AdjointSolve = false
    while it <= nt

        AdjointSolve = adjoint && it == nt
        step_η_multiplier = it == nt ? η_multiplier : nothing
        # Stokes solver ----------------
        args = (; T = @ones(ni .+ 2...), P = stokes.P, dt = dt, ΔT = @zeros(ni .+ 2...))
        solve_DYREL!(
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
            viscosity_cutoff = viscosity_cutoff,
            # a cell-wise multiplier only survives if the τII viscosity refresh is switched off
            linear_viscosity = !isnothing(η_multiplier),
            η_multiplier = step_η_multiplier,
            adjoint = AdjointSolve,
            observation = observation,
        )
        dt = compute_dt(stokes, di, igg) * 0.1
        # ------------------------------

        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        velocity = @. √(Vx_v^2 + Vy_v^2)

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), ())
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)


        if plot_results && it in (0, nt)
            # Plotting ---------------------
            # adjoint velocities live on the same staggered grid as the forward ones,
            # so they get interpolated to the vertices the same way
            λVx_v = @zeros(ni .+ 1...)
            λVy_v = @zeros(ni .+ 1...)
            velocity2vertex!(λVx_v, λVy_v, stokes_ad.λV.Vx, stokes_ad.λV.Vy)

            xc = xci
            xv = xvi
            ρ = Array(ρg[2])

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

            # forward Stokes solution
            panel!(1, 1, "ρ", xc, ρ; colormap = :roma)
            panel!(1, 2, "Vx", xv, Vx_v; colormap = :vik, diverging = true)
            panel!(1, 3, "Vy", xv, Vy_v; colormap = :vik, diverging = true)
            panel!(1, 4, "P", xc, stokes.P; colormap = :batlow)

            # adjoint solution
            panel!(2, 1, "λVx", xv, λVx_v; colormap = :vik, diverging = true)
            panel!(2, 2, "λVy", xv, λVy_v; colormap = :vik, diverging = true)
            panel!(2, 3, "ρ (adjoint)", xc, stokes_ad.ρ; colormap = :roma)
            panel!(2, 4, "η (adjoint)", xc, stokes_ad.viscosity.η; colormap = :batlow)

            save(
                joinpath(figdir, "$(it).png"),
                f
            )
            display(f)
        end
        it += 1
        verbose && @show extrema(stokes.∇V)
        # ------------------------------
    end
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    cost = sum(@view(stokes.V.Vy[mask.i, mask.j]))
    return return_fields ? (;
            cost,
            viscosity_gradient = Array(stokes_ad.viscosity.η),
            viscosity_gradient_vertex = Array(stokes_ad.viscosity.ηv),
            viscosity = Array(stokes.viscosity.η),
            viscosity_vertex = Array(stokes.viscosity.ηv),
            viscosity_direction = Array(stokes.viscosity.η),
            viscosity_direction_vertex = Array(stokes.viscosity.ηv),
        ) : nothing
end

# Define `NO_AUTORUN` before including this file to load the functions without running the
# demo (see ViscosityGradientTest.jl). A plain `include` from the REPL still runs it.
if !@isdefined(NO_AUTORUN)
    ar = 1 # aspect ratio
    n = 1
    nx = 32 * n
    ny = 32 * n
    figdir = "SinkingBlock2D_VE_adjoint"
    # A global grid may still be active from an earlier run in this session, which makes
    # `init_global_grid` throw. Tear it down first, keeping MPI alive so it can be re-created.
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    sinking_block2D_VE(igg; ar = ar, nx = nx, ny = ny, figdir = figdir)
end
