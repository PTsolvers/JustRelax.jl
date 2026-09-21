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
using GeoParams, Printf, Random
using CairoMakie:
    Axis,
    Colorbar,
    DataAspect,
    Figure,
    Label,
    heatmap!,
    hidexdecorations!,
    hideydecorations!,
    save

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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

# Material parameters this script differentiates. `:G` is a multiplier-based control on the
# shear modulus; the rest are parameters of the `PT_Density` model
#
#   ρ(T, P) = ρ0 * (1 - α * (T - T0) + β * (P - P0))
#
# and come back resolved per phase. The viscosity sensitivity needs no control at all: the
# adjoint always fills `stokes_ad.viscosity.η` / `.ηv`.
const DENSITY_PARAMETERS = (:ρ0, :α, :β, :T0, :P0)
const MATERIAL_PARAMETERS = (:G, DENSITY_PARAMETERS...)

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function sinking_block2D_VE_materials(
        igg;
        ar = 8,
        ny = 16,
        nx = ny * 8,
        figdir = "SinkingBlock2D_VE_adjoint_materials",
        nt = 10,
        # material parameter perturbations, used by the finite-difference verification
        δ = (;),
        ΔT_amplitude = 0.05,
        return_fields = false,
        plot_results = true,
        rngseed = 1234,
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
    # `δ` perturbs one parameter of one phase, so the gradients can be checked against a
    # finite difference of the cost without touching the rest of the setup.
    perturb(phase, name, value) = value + get(get(δ, phase, (;)), name, 0.0)

    η_mantle = 1.0
    η_block = 10.0
    gravity = 1.0
    G_base = 0.1
    G = perturb(:both, :G, G_base)
    # PT_Density parameters. T0 = 0 and P0 = 0 keep (T - T0) and (P - P0) away from zero,
    # so none of the five density gradients vanishes for a trivial reason.
    mantle_density = (;
        ρ0 = perturb(:mantle, :ρ0, 1.0),
        α = perturb(:mantle, :α, 0.1),
        β = perturb(:mantle, :β, 0.05),
        T0 = perturb(:mantle, :T0, 0.0),
        P0 = perturb(:mantle, :P0, 0.0),
    )
    block_density = (;
        ρ0 = perturb(:block, :ρ0, 1.2),
        α = perturb(:block, :α, 0.2),
        β = perturb(:block, :β, 0.1),
        T0 = perturb(:block, :T0, 0.0),
        P0 = perturb(:block, :P0, 0.0),
    )

    elasticity = ConstantElasticity(; G = G, Kb = 5G)
    rheology = (
        SetMaterialParams(;
            Name = "Mantle",
            Phase = 1,
            Density = PT_Density(; mantle_density...),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_mantle), elasticity)),
            Elasticity = elasticity,
            Gravity = ConstantGravity(; g = gravity),
        ),
        SetMaterialParams(;
            Name = "Block",
            Phase = 2,
            Density = PT_Density(; block_density...),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_block), elasticity)),
            Elasticity = elasticity,
            Gravity = ConstantGravity(; g = gravity),
        ),
    )
    # One mantle Maxwell time keeps both viscous and elastic deformation active. It is tied
    # to the *unperturbed* G: the adjoint differentiates the residual at a fixed time step,
    # so letting δG move dt too would put a term into the finite difference that the
    # gradient does not contain.
    dt = η_mantle / G_base
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    # `init_particles` jitters the particle positions, so two runs of the same setup give
    # slightly different phase ratios and hence slightly different costs. That difference
    # swamps a finite difference taken with a small step, so the seed is fixed here.
    Random.seed!(rngseed)
    nxcell, max_xcell, min_xcell = 40, 40, 12
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    # Rectangular density anomaly
    xc_anomaly = 0.0      # centred horizontally (domain spans -lx/2 .. lx/2)
    yc_anomaly = -ly / 4  # centred vertically; `init_phases!` takes it as a depth
    r_anomaly_x = 0.2
    r_anomaly_y = 0.1
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    # location of the block
    xc_anomaly_block = 0.0
    yc_anomaly_block = -ly / 2
    r_anomaly_block = 0.1
    init_phases!(pPhases, particles, xc_anomaly_block, abs(yc_anomaly_block), r_anomaly_block)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend, ni)

    # Adjoint Stokes --------------------------------------
    stokes_ad = AdjointStokesArrays(backend, ni)
    # `:G` gets center/vertex multiplier fields; the density parameters only get gradient
    # buffers of size (nphases, ni...), hence `nphases`.
    controls, gradients = material_controls(
        backend, ni, MATERIAL_PARAMETERS; nphases = length(rheology)
    )

    # Temperature and its per-step change. A non-zero ΔT is what activates the second
    # route into α: the thermal term α * ΔT / dt of the pressure residual. With ΔT = 0
    # only the buoyancy route ρ(α, T, P) contributes.
    T = @ones(ni .+ 2...)
    ΔT = @fill(ΔT_amplitude, ni .+ 2...)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = T, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args = (; T = T, P = stokes.P, dt = dt)
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

        AdjointSolve = it == nt
        step_controls = it == nt ? controls : (;)
        step_gradients = it == nt ? gradients : nothing
        # Stokes solver ----------------
        args = (; T = T, P = stokes.P, dt = dt, ΔT = ΔT)
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
            adjoint = AdjointSolve,
            observation = observation,
            controls = step_controls,
            gradients = step_gradients,
        )
        dt = compute_dt(stokes, di, igg) * 0.1
        # ------------------------------

        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

        # Advection --------------------
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)

        if plot_results && AdjointSolve
            plot_sensitivities(
                figdir, it, grid, stokes, stokes_ad, gradients, ρg, Vx_v, Vy_v
            )
        end
        it += 1
        verbose && @show extrema(stokes.∇V)
        # ------------------------------
    end

    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    cost = sum(@view(stokes.V.Vy[mask.i, mask.j]))

    material_gradients = (;
        # per-cell fields
        η = Array(stokes_ad.viscosity.η),
        ηv = Array(stokes_ad.viscosity.ηv),
        ρ = Array(stokes_ad.ρ),
        G = Array(gradients.G.center),
        # per-phase fields, size (nphases, ni...)
        map(name -> name => Array(gradients[name].center), DENSITY_PARAMETERS)...,
    )
    # The same parameters as single spatial fields: each phase's contribution is already
    # weighted by its phase ratio, so summing over the phase axis is the sensitivity to a
    # change of the parameter in *every* phase at once, cell by cell.
    merged_gradients = merge_phases(material_gradients)
    verbose && report_gradients(material_gradients, length(rheology))

    return if return_fields
        (; cost, gradients = material_gradients, merged_gradients, shear_modulus = G)
    else
        nothing
    end
end

"""
    merge_phases(g)

Collapse the per-phase density gradients of `g` (each of size `(nphases, ni...)`) onto a
single spatial field per parameter by summing over the phase axis.
"""
function merge_phases(g)
    return NamedTuple{DENSITY_PARAMETERS}(
        map(name -> merge_phases(g[name]), DENSITY_PARAMETERS)
    )
end

merge_phases(field::AbstractArray) = dropdims(sum(Array(field); dims = 1); dims = 1)

# Scalar sensitivity of the cost to a uniform change of one parameter: the per-cell
# derivatives simply add up, because a uniform perturbation hits every cell at once.
total_gradient(field) = sum(field)
total_gradient(field, phase) = sum(@view field[phase, ntuple(_ -> :, ndims(field) - 1)...])

function report_gradients(g, nphases)
    @printf("\n######## Material sensitivities (dJ/dp for a uniform p) ########\n")
    @printf("  %-14s %+1.6e\n", "η (all cells)", total_gradient(g.η))
    @printf("  %-14s %+1.6e\n", "ρ (all cells)", total_gradient(g.ρ))
    @printf("  %-14s %+1.6e\n", "G", total_gradient(g.G))
    for phase in 1:nphases, name in DENSITY_PARAMETERS
        @printf("  %-14s %+1.6e\n", "$name (phase $phase)", total_gradient(g[name], phase))
    end
    return nothing
end

function plot_sensitivities(figdir, it, grid, stokes, stokes_ad, gradients, ρg, Vx_v, Vy_v)
    xc, xv = grid.xci, grid.xvi

    # adjoint velocities live on the same staggered grid as the forward ones
    ni = size(stokes.P)
    λVx_v = @zeros(ni .+ 1...)
    λVy_v = @zeros(ni .+ 1...)
    velocity2vertex!(λVx_v, λVy_v, stokes_ad.λV.Vx, stokes_ad.λV.Vy)

    # a constant field (an all-zero gradient, say) has no colour range Makie can use
    function crange(A, diverging)
        if diverging
            m = maximum(abs, A)
            return iszero(m) ? (-1.0, 1.0) : (-m, m)
        end
        lo, hi = extrema(A)
        return lo == hi ? (lo - 1, hi + 1) : (lo, hi)
    end

    nrows, ncols = 3, 5
    f = Figure(size = (400 * ncols, 320 * nrows))
    function panel!(row, col, title, x, A; colormap = :batlow, diverging = false)
        ax = Axis(
            f[row, 2col - 1];
            aspect = DataAspect(), title = title,
            xlabel = row == nrows ? "x" : "", ylabel = col == 1 ? "y" : "",
        )
        h = heatmap!(ax, x..., Array(A); colormap = colormap, colorrange = crange(Array(A), diverging))
        Colorbar(f[row, 2col], h)
        row < nrows && hidexdecorations!(ax, grid = false)
        col > 1 && hideydecorations!(ax, grid = false)
        return nothing
    end

    # forward Stokes solution
    panel!(1, 1, "ρg_y", xc, ρg[2]; colormap = :roma)
    panel!(1, 2, "Vx", xv, Vx_v; colormap = :vik, diverging = true)
    panel!(1, 3, "Vy", xv, Vy_v; colormap = :vik, diverging = true)
    panel!(1, 4, "P", xc, stokes.P)
    panel!(1, 5, "η", xc, stokes.viscosity.η)

    # adjoint state and the gradients that need no per-phase resolution
    panel!(2, 1, "λVx", xv, λVx_v; colormap = :vik, diverging = true)
    panel!(2, 2, "λVy", xv, λVy_v; colormap = :vik, diverging = true)
    # dJ/dη is what the adjoint fills directly; dJ/dρ is the cell-wise density
    # sensitivity the per-phase parameter gradients are built from.
    panel!(2, 3, "∂J/∂η", xc, stokes_ad.viscosity.η; colormap = :vik, diverging = true)
    panel!(2, 4, "∂J/∂ρ", xc, stokes_ad.ρ; colormap = :vik, diverging = true)
    # the center field is the complete dJ/dG: it includes the vertex contribution
    # pulled back through center2vertex!
    panel!(2, 5, "∂J/∂G", xc, gradients.G.center; colormap = :vik, diverging = true)

    # density-parameter gradients, summed over the phase axis: one spatial field each,
    # the sensitivity to changing that parameter in every phase at once
    for (col, name) in enumerate(DENSITY_PARAMETERS)
        panel!(
            3, col, "∂J/∂$name", xc, merge_phases(gradients[name].center);
            colormap = :vik, diverging = true,
        )
    end

    save(joinpath(figdir, "$(it).png"), f)
    return f
end

"""
    verify_gradient(igg, phase, name; h = 1.0e-5, kwargs...)

Central finite difference of the cost with respect to a single material parameter,
against the adjoint gradient of the same parameter. `phase` is `:mantle`, `:block`, or
`:both` for the shared shear modulus `:G`.
"""
function verify_gradient(igg, phase::Symbol, name::Symbol; h = 1.0e-5, nt = 1, kwargs...)
    base = sinking_block2D_VE_materials(
        igg; nt = nt, return_fields = true, plot_results = false, verbose = false, kwargs...
    )
    step(sign) = sinking_block2D_VE_materials(
        igg; nt = nt, return_fields = true, plot_results = false, verbose = false,
        δ = NamedTuple{(phase,)}((NamedTuple{(name,)}((sign * h,)),)), kwargs...
    ).cost

    finite_difference = (step(+1) - step(-1)) / (2h)
    adjoint_gradient = if name === :G
        total_gradient(base.gradients.G)
    else
        total_gradient(base.gradients[name], phase === :mantle ? 1 : 2)
    end
    @printf(
        "%-6s phase %-7s adjoint = %+1.6e   FD = %+1.6e   rel. err = %1.3e\n",
        name, phase, adjoint_gradient, finite_difference,
        abs(adjoint_gradient - finite_difference) / max(abs(finite_difference), eps())
    )
    return (; adjoint = adjoint_gradient, finite_difference)
end

# Define NO_AUTORUN before including this file to load the functions without running.
if !@isdefined(NO_AUTORUN)
    ar = 1 # aspect ratio
    n = 1
    nx = 32 * n
    ny = 32 * n
    figdir = "SinkingBlock2D_VE_adjoint_materials"
    # A global grid may still be active from an earlier run in this session, which makes
    # `init_global_grid` throw. Tear it down first, keeping MPI alive so it can be re-created.
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    sinking_block2D_VE_materials(igg; ar = ar, nx = nx, ny = ny, figdir = figdir)
end
