const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDA.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
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
import JustPIC.GridGeometryUtils as GGU

using GeoParams
using CairoMakie

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function init_phases!(phases, particles)
    ni = size(phases)

    radius = 100.0e3
    origin = 250.0e3, 250.0e3
    circle = GGU.Circle(origin, radius)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if 0.0e0 ≤ depth ≤ 100.0e3
                @index phases[ip, i, j] = 1.0

            else
                @index phases[ip, i, j] = 2.0
                p = GGU.Point(x, depth)
                if GGU.inside(p, circle)
                    @index phases[ip, i, j] = 3.0
                end
            end

        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end

# density and viscosity of air, mantle and plume
plume_parameters() = ((ρ = 1.0e1, η = 1.0e17), (ρ = 3.3e3, η = 1.0e21), (ρ = 3.2e3, η = 1.0e20))

function plume_rheology(parameters = plume_parameters())
    return ntuple(length(parameters)) do p
        SetMaterialParams(;
            Phase = p,
            Density = ConstantDensity(; ρ = parameters[p].ρ),
            CompositeRheology = CompositeRheology((LinearViscous(; η = parameters[p].η),)),
            Gravity = ConstantGravity(; g = 9.81),
        )
    end
end

# value of a material parameter of phase `p`, as stored in the rheology
material_parameter(rheology, p, ::Val{:ρ}) = rheology[p].Density[1].ρ.val
material_parameter(rheology, p, ::Val{:η}) = rheology[p].CompositeRheology[1].elements[1].η.val

# Objective of the adjoint solve: the mean rise rate of the plume, J = Σ w·Vy, with w the plume
# fraction at the Vy nodes normalized to sum to one. Built from the current phase ratios, so the
# objective follows the plume as it rises.
function plume_observation(phase_ratios, plume_phase, Vy)
    weights = zeros(size(Vy))
    ratios = phase_ratios.Vy
    for j in axes(ratios, 2), i in axes(ratios, 1)
        # the Vy array carries one ghost column on either side
        weights[i + 1, j] = ratios[i, j][plume_phase]
    end
    weights ./= sum(weights)
    return (; field = :Vy, weights)
end

kyr(t) = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)

# Velocity arrows at every `stride`-th vertex.
function draw_velocity!(ax, stokes, xvi; stride = 5, color = :red)
    Vx_v = @zeros(size(stokes.P) .+ 1...)
    Vy_v = @zeros(size(stokes.P) .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    arrows2d!(
        ax,
        xvi[1][1:stride:(end - 1)] ./ 1.0e3, xvi[2][1:stride:(end - 1)] ./ 1.0e3, Array.((Vx_v[1:stride:(end - 1), 1:stride:(end - 1)], Vy_v[1:stride:(end - 1), 1:stride:(end - 1)]))...,
        lengthscale = 25 / max(maximum(Vx_v), maximum(Vy_v)),
        color = color,
    )
    return ax
end

# Particle phases, velocity arrows and the free-surface marker chain.
function draw_forward!(ax, particles, pPhases, chain, stokes, xvi)
    # Make particles plottable
    ppx, ppy = particles.coords
    pxv = ppx.data[:] ./ 1.0e3
    pyv = ppy.data[:] ./ 1.0e3
    clr = pPhases.data[:]
    idxv = particles.index.data[:]

    chain_x = chain.coords[1].data[:] ./ 1.0e3
    chain_y = chain.coords[2].data[:] ./ 1.0e3

    scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 5)
    draw_velocity!(ax, stokes, xvi)
    scatter!(ax, Array(chain_x), Array(chain_y), color = :red, markersize = 5)
    return ax
end

# Forward state and sensitivities of the step the adjoint differentiates, in one figure: the
# forward solution (phases, velocity, free surface), the viscosity overlaid by the velocity, and
# the spatial density and viscosity sensitivities. The viscosity sensitivity is shown as
# η·∂J/∂η, the response to a relative viscosity perturbation. The outline of the plume that J
# observes (half the maximum weight) is drawn in every panel.
function plot_adjoint(figdir, it, t, particles, pPhases, chain, stokes, stokes_ad, ϕ, observation, xci, xvi)
    # blank the air: cells without rock
    rock = Array(ϕ.center) .> 0
    masked(A) = ifelse.(rock, Array(A), NaN)
    xc_km, xv_km = xci ./ 1.0e3, xvi ./ 1.0e3
    domain = (extrema(xv_km[1])..., extrema(xv_km[2])...)

    function symmetric_range(A)
        m = maximum(x -> isnan(x) ? zero(x) : abs(x), A)
        return iszero(m) ? (-1.0, 1.0) : (-m, m)
    end

    # objective weights at the Vy nodes, without the ghost columns
    weights = Array(observation.weights)[2:(end - 1), :]
    function finish!(ax)
        contour!(ax, xc_km[1], xv_km[2], weights; levels = [0.5 * maximum(weights)], color = :black)
        limits!(ax, domain...)
        return ax
    end

    fig = Figure(size = (1200, 1100))
    Label(fig[0, 1:4], "t = $(kyr(t)) Kyrs"; fontsize = 20, font = :bold)

    ax = Axis(fig[1, 1]; aspect = 1, title = "forward: phases, velocity, free surface", ylabel = "y [km]")
    draw_forward!(ax, particles, pPhases, chain, stokes, xvi)
    finish!(ax)

    ax = Axis(fig[1, 3]; aspect = 1, title = "log10 η [Pa s] and velocity")
    h = heatmap!(ax, xc_km..., log10.(masked(stokes.viscosity.η)); colormap = :viridis)
    draw_velocity!(ax, stokes, xvi; color = :white)
    finish!(ax)
    Colorbar(fig[1, 4], h)

    panels = (
        (1, "∂J/∂ρ", masked(stokes_ad.ρ)),
        (3, "η ∂J/∂η", masked(stokes_ad.viscosity.η .* stokes.viscosity.η)),
    )
    for (col, title, A) in panels
        ax = Axis(fig[2, col]; aspect = 1, title, xlabel = "x [km]", ylabel = col == 1 ? "y [km]" : "")
        h = heatmap!(ax, xc_km..., A; colormap = :vik, colorrange = symmetric_range(A))
        finish!(ax)
        Colorbar(fig[2, col + 1], h)
    end
    save(joinpath(figdir, "adjoint_$(it).png"), fig)
    return fig
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
"""
    plume_free_surface_adjoint(igg, nx, ny; nt = 30, adjoint_every = 6, kwargs...)

Rising plume below a free surface, solved with the variational DYREL solver as in
`miniapps/DYREL2D/free_surface_stabilization/PlumeFreeSurface_VariationalDYREL.jl`. Every
`adjoint_every`-th time step the adjoint solver computes the sensitivity of the mean rise rate of
the plume,

    J = Σ w·Vy,   w = plume fraction at the Vy nodes, normalized to Σ w = 1,

to the density `ρ` and viscosity `η` of every phase (air, mantle, plume), and saves the forward
state and the sensitivities in one figure. The weights are built from the phase ratios of that
step, so the objective follows the plume wherever it has risen to. The rock ratio ϕ and the
marker chain are frozen during the adjoint solve.

Returns, with `return_fields = true`, `J` of the last step (`cost`), the sensitivities of the
last adjoint step, and the history `(; step, t, J)` of every adjoint step.

For Taylor tests (`benchmark/TaylorTests2D_variational.jl`): `final_parameters` (see
`plume_parameters`) and `η_multiplier` perturb only the last step, the one the adjoint
differentiates, and `adjoint = false` skips every adjoint solve.
"""
function plume_free_surface_adjoint(
        igg, nx, ny;
        nt = 30,
        adjoint_every = 6,
        figdir = "PlumeFreeSurface2D_adjoint",
        solver_ϵ = 1.0e-6,
        solver_ϵ_vel = 1e-2,
        CFL = 0.99,
        c_fact = 0.5,
        # the free surface admits a uniform volumetric pressure mode that the Powell-Hestenes
        # loop removes only slowly with the default penalty (γfact = 20)
        γfact = 100.0,
        iterMax = 100.0e3,
        plot_results = true,
        return_fields = false,
        verbose = true,
        adjoint = true,
        parameters = plume_parameters(),
        final_parameters = nothing,
        η_multiplier = nothing,
    )

    # Physical domain ------------------------------------
    thick_air = 100.0e3             # thickness of sticky air layer
    ly = 400.0e3 + thick_air # domain length in y
    lx = 500.0e3             # domain length in x
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = plume_rheology(parameters)
    final_rheology = isnothing(final_parameters) ? rheology : plume_rheology(final_parameters)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 15
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    grid_vxi = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Initialize marker chain-------------------------------
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = -100.0e3
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    # ----------------------------------------------------

    # rock ratios for variational stokes
    # RockRatios
    air_phase = 1
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, di)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # ----------------------------------------------------

    # ADJOINT --------------------------------------------
    # Adjoint arrays and the material parameters to differentiate with respect to. Each name
    # collects every use of that parameter, phase by phase.
    stokes_ad = AdjointStokesArrays(backend, ni)
    target_parameters = (:ρ, :η)
    gradients = material_controls(backend, ni, target_parameters; nphases = length(rheology))
    # Objective: mean rise rate of the plume material, built at the adjoint step (see
    # `plume_observation`)
    plume_phase = 3
    observation = nothing
    cost = NaN
    history = (; step = Int[], t = Float64[], J = Float64[])
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    compute_lithostatic_pressure!(stokes.P, ρg[2], di[2], igg)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf); air_phase = air_phase)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false
    )

    plot_results && take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = 10.0e3 * (3600 * 24 * 365.25)
    viscosity_cutoff = (-Inf, Inf)
    dyrel = DYREL(
        backend,
        stokes,
        rheology,
        phase_ratios,
        ϕ,
        grid.di,
        dt;
        ϵ = solver_ϵ,
        ϵ_vel = solver_ϵ_vel,
        CFL,
        c_fact,
        γfact,
    )

    while it < nt
        last_step = it == nt - 1
        # the adjoint linearizes around the converged forward state of this step
        run_adjoint = adjoint && iszero(rem(it + 1, adjoint_every))
        # weight the objective by where the plume is now
        (run_adjoint || last_step) &&
            (observation = plume_observation(phase_ratios, plume_phase, stokes.V.Vy))
        step_rheology = last_step ? final_rheology : rheology

        # Stokes -----------------------
        result = solve_VariationalDYREL!(
            stokes,
            stokes_ad,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ,
            step_rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                air_phase = air_phase,
                iterMax = iterMax,
                total_iterMax = iterMax,
                viscosity_relaxation = 1.0e-2,
                nout = 2.0e3,
                viscosity_cutoff = viscosity_cutoff,
                free_surface = true,
                # the adjoint differentiates the stress update with a fixed viscosity; the
                # rheology is linear, so this does not change the forward solution
                linear_viscosity = true,
                adjoint = run_adjoint,
                observation = observation,
                gradients = gradients,
                η_multiplier = last_step ? η_multiplier : nothing,
                verbose_PH = verbose,
                verbose_DR = false,
            ),
        )
        result.converged || error("Variational DYREL did not converge (err=$(result.err))")

        if run_adjoint || last_step
            cost = sum(observation.weights .* Array(stokes.V.Vy))
        end
        if run_adjoint
            push!(history.step, it + 1); push!(history.t, t); push!(history.J, cost)
            println("\nStep $(it + 1): plume rise rate J = $(cost) m/s")
            println("Sensitivity of J to the material parameters")
            println("(p ∂J/∂p: change of J for a 100 % change of p, summed over the domain)")
            for name in target_parameters, p in eachindex(rheology)
                value = material_parameter(step_rheology, p, Val(name))
                total = sum(Array(gradients[name].center)[p, :, :])
                println("  phase $p  $name = $(value):  p ∂J/∂p = $(value * total)")
            end
            plot_results && plot_adjoint(figdir, it + 1, t, particles, pPhases, chain, stokes, stokes_ad, ϕ, observation, xci, xvi)
        end

        dt = compute_dt(stokes, di) * 0.95
        println("t = $(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs, dt = $(round(dt / (3600 * 24 * 365.25); digits = 3)) yrs")
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)

        # Filter against the new surface before injection.
        semilagrangian_advection_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, xvi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), ())
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # the adjoint uses the rock ratio of the step it differentiates, so it is updated here,
        # after the adjoint solve
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)
        # ------------------------------

        @show it += 1
        t += dt
    end

    return return_fields ? (;
            cost,
            history,
            phase_gradients = map(gradient -> Array(gradient.center), gradients),
            density_gradient = Array(stokes_ad.ρ),
            viscosity_gradient = Array(stokes_ad.viscosity.η),
            viscosity_gradient_vertex = Array(stokes_ad.viscosity.ηv),
            viscosity = Array(stokes.viscosity.η),
            viscosity_vertex = Array(stokes.viscosity.ηv),
            rock_ratio = Array(ϕ.center),
        ) : nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# Define `NO_AUTORUN` before including this file to load the functions without running the
# demo. A plain `include` from the REPL still runs it.
if !@isdefined(NO_AUTORUN)
    n = 64
    nx = n
    ny = n
    # A global grid may still be active from an earlier run in this session, which makes
    # `init_global_grid` throw. Tear it down first, keeping MPI alive so it can be re-created.
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    plume_free_surface_adjoint(igg, nx, ny)
end
