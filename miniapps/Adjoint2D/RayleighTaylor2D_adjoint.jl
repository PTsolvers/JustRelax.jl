const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDA.CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

using GeoParams
using CairoMakie

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)
        interface(x) = A * sinpi(x / 500.0e3)

        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue
            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            phase = if 0.0 ≤ depth ≤ 100.0e3
                1.0
            elseif depth > -interface(x) + (200.0e3 - A)
                3.0
            else
                2.0
            end
            @index phases[ip, i, j] = phase
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
end

rayleigh_taylor_parameters() = (
    (ρ = 0.0, η = 1.0e16),
    (ρ = 3.3e3, η = 1.0e21),
    (ρ = 3.2e3, η = 1.0e20),
)

function rayleigh_taylor_rheology(parameters = rayleigh_taylor_parameters())
    return ntuple(length(parameters)) do phase
        SetMaterialParams(;
            Phase = phase,
            Density = ConstantDensity(; ρ = parameters[phase].ρ),
            CompositeRheology = CompositeRheology((LinearViscous(; η = parameters[phase].η),)),
            Gravity = ConstantGravity(; g = 9.81),
        )
    end
end

material_parameter(rheology, phase, ::Val{:ρ}) = rheology[phase].Density[1].ρ.val
material_parameter(rheology, phase, ::Val{:η}) =
    rheology[phase].CompositeRheology[1].elements[1].η.val

# Mean vertical velocity in a fixed box on the Vy grid.
function box_observation(grid, Vy, center, half_width)
    weights = zeros(size(Vy))
    i = findall(x -> abs(x - center[1]) <= half_width[1], grid.xci[1])
    j = findall(y -> abs(y - center[2]) <= half_width[2], grid.xvi[2])
    (isempty(i) || isempty(j)) &&
        throw(ArgumentError("the observation box does not contain any Vy nodes"))
    weights[i .+ 1, j] .= 1
    weights ./= sum(weights)
    return (; field = :Vy, weights)
end

kyr(t) = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)

function draw_velocity!(ax, stokes, xvi; stride = 5, color = :red)
    Vxv = @zeros(size(stokes.P) .+ 1...)
    Vyv = @zeros(size(stokes.P) .+ 1...)
    velocity2vertex!(Vxv, Vyv, @velocity(stokes)...)
    Vmax = max(maximum(abs, Vxv), maximum(abs, Vyv))
    iszero(Vmax) && return ax
    arrows2d!(
        ax,
        xvi[1][1:stride:(end - 1)] ./ 1.0e3,
        xvi[2][1:stride:(end - 1)] ./ 1.0e3,
        Array(Vxv[1:stride:(end - 1), 1:stride:(end - 1)]),
        Array(Vyv[1:stride:(end - 1), 1:stride:(end - 1)]);
        lengthscale = 25 / Vmax,
        color,
    )
    return ax
end

function draw_forward!(ax, particles, pPhases, chain, stokes, xvi)
    px, py = particles.coords
    active = particles.index.data[:] .!= 0
    scatter!(
        ax,
        Array(px.data[:][active]) ./ 1.0e3,
        Array(py.data[:][active]) ./ 1.0e3;
        color = Array(pPhases.data[:][active]),
        colormap = :grayC,
        markersize = 4,
    )
    draw_velocity!(ax, stokes, xvi)
    scatter!(
        ax,
        Array(chain.coords[1].data[:]) ./ 1.0e3,
        Array(chain.coords[2].data[:]) ./ 1.0e3;
        color = :red,
        markersize = 4,
    )
    return ax
end

function plot_adjoint(
        figdir, step, t, particles, pPhases, chain, stokes, stokes_ad, ρ, ϕ,
        observation, xci, xvi,
    )
    rock = Array(ϕ.center) .> 0
    masked(A) = ifelse.(rock, Array(A), NaN)
    xckm, xvkm = xci ./ 1.0e3, xvi ./ 1.0e3
    domain = (extrema(xvkm[1])..., extrema(xvkm[2])...)

    symmetric_range(A) = begin
        limit = maximum(x -> isnan(x) ? zero(x) : abs(x), A)
        iszero(limit) ? (-1.0, 1.0) : (-limit, limit)
    end

    weights = Array(observation.weights)[2:(end - 1), :]
    weight_max = maximum(weights)
    function finish!(ax)
        heatmap!(
            ax, xckm[1], xvkm[2], weights;
            colormap = [RGBAf(1, 0, 1, 0), RGBAf(1, 0, 1, 0.22)],
            colorrange = (0, weight_max),
        )
        contour!(
            ax, xckm[1], xvkm[2], weights;
            levels = [0.5 * weight_max], color = :magenta, linewidth = 3,
        )
        limits!(ax, domain...)
        return ax
    end

    fig = Figure(size = (1200, 1100))
    Label(
        fig[0, 1:4], "step $step, t = $(kyr(t)) kyr; magenta = observation region";
        fontsize = 20, font = :bold,
    )

    ax = Axis(fig[1, 1]; aspect = 1, title = "forward state", ylabel = "y [km]")
    draw_forward!(ax, particles, pPhases, chain, stokes, xvi)
    finish!(ax)

    ax = Axis(fig[1, 3]; aspect = 1, title = "log10 eta [Pa s] and velocity")
    h = heatmap!(ax, xckm..., log10.(masked(stokes.viscosity.η)); colormap = :viridis)
    draw_velocity!(ax, stokes, xvi; color = :white)
    finish!(ax)
    Colorbar(fig[1, 4], h)

    panels = (
        (1, "ρ ∂J/∂ρ", masked(ρ .* stokes_ad.ρ)),
        (3, "η ∂J/∂η", masked(stokes_ad.viscosity.η .* stokes.viscosity.η)),
    )
    for (column, title, values) in panels
        ax = Axis(
            fig[2, column]; aspect = 1, title, xlabel = "x [km]",
            ylabel = column == 1 ? "y [km]" : "",
        )
        h = heatmap!(ax, xckm..., values; colormap = :vik, colorrange = symmetric_range(values))
        finish!(ax)
        Colorbar(fig[2, column + 1], h)
    end

    save(joinpath(figdir, "adjoint_$step.png"), fig)
    return fig
end

"""
    rayleigh_taylor2D_adjoint(igg, nx, ny; kwargs...)

Variational-DYREL Rayleigh-Taylor example with a free surface. At selected steps, the
adjoint computes sensitivities of the mean vertical velocity in a small central box within
the lithosphere phase,

    J = sum(w .* Vy),

where `w` is uniform in the observation box and zero elsewhere. The sensitivity figure
contains the forward state, viscosity, `rho*dJ/drho`, and `eta*dJ/deta`.
"""
function rayleigh_taylor2D_adjoint(
        igg, nx, ny;
        nt = 10,
        adjoint_every = 6,
        figdir = "RayleighTaylor2D_adjoint",
        solver_ϵ = 1.0e-6,
        solver_ϵ_vel = 1.0e-2,
        CFL = 0.99,
        c_fact = 0.5,
        γfact = 100.0,
        iterMax = 150.0e3,
        nout = 2.0e3,
        observation_center = (250.0e3, -150.0e3),
        observation_half_width = (25.0e3, 20.0e3),
        plot_results = true,
        return_fields = false,
        verbose = true,
        adjoint = true,
    )
    thick_air = 100.0e3
    lx, ly = 500.0e3, 500.0e3 + thick_air
    ni = nx, ny
    li = lx, ly
    di = li ./ ni
    origin = 0.0, -ly
    grid = Geometry(ni, li; origin)
    (; xci, xvi) = grid
    grid_vxi = velocity_grids(xci, xvi, di)

    rheology = rayleigh_taylor_rheology()

    nxcell, max_xcell, min_xcell = 192, 256, 64
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)
    init_phases!(pPhases, particles, 5.0e3)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    nxcell, max_xcell, min_xcell = 100, 200, 15
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], -100.0e3)

    air_phase = 1
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    target_parameters = (:ρ, :η)
    gradients = material_controls(backend, ni, target_parameters; nphases = length(rheology))
    observation = box_observation(
        grid, stokes.V.Vy, observation_center, observation_half_width
    )
    cost = NaN
    history = (; step = Int[], t = Float64[], J = Float64[])

    thermal = ThermalArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase)
    ρg_rock = ρg[2] .* ϕ.center
    compute_lithostatic_pressure!(stokes.P, ρg_rock, di[2], igg)
    @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0)
    viscosity_cutoff = (-Inf, Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        free_surface = false,
    )

    plot_results && take(figdir)

    t, step = 0.0, 0
    dt = 10.0e3 * (3600 * 24 * 365.25)
    dt_max = 25.0e3 * (3600 * 24 * 365.25)
    dyrel = DYREL(
        backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt;
        ϵ = solver_ϵ,
        ϵ_vel = solver_ϵ_vel,
        CFL,
        c_fact,
        γfact,
    )

    while step < nt
        last_step = step == nt - 1
        run_adjoint = adjoint && (last_step || iszero(rem(step + 1, adjoint_every)))

        result = solve_VariationalDYREL!(
            stokes,
            stokes_ad,
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
                air_phase,
                iterMax,
                total_iterMax = iterMax,
                viscosity_relaxation = 1.0e-2,
                nout,
                free_surface = true,
                viscosity_cutoff,
                linear_viscosity = true,
                adjoint = run_adjoint,
                observation,
                gradients,
                verbose_PH = verbose,
                verbose_DR = false,
            ),
        )
        result.converged || error("Variational DYREL did not converge (err=$(result.err))")

        if run_adjoint || last_step
            cost = sum(observation.weights .* Array(stokes.V.Vy))
        end
        if run_adjoint
            push!(history.step, step + 1)
            push!(history.t, t)
            push!(history.J, cost)
            println("\nStep $(step + 1): observation-box mean Vy J = $cost m/s")
            println("Sensitivity of J to the material parameters")
            for name in target_parameters, phase in eachindex(rheology)
                value = material_parameter(rheology, phase, Val(name))
                total = sum(Array(gradients[name].center)[phase, :, :])
                println("  phase $phase  $name = $value: p*dJ/dp = $(value * total)")
            end
            if plot_results
                ρ = ρg[2] ./ compute_gravity(first(rheology))
                plot_adjoint(
                    figdir, step + 1, t, particles, pPhases, chain, stokes, stokes_ad, ρ, ϕ,
                    observation, xci, xvi,
                )
            end
        end

        dt = compute_dt(stokes, di, dt_max)
        println("t = $(kyr(t)) kyr, dt = $(dt / (3600 * 24 * 365.25)) yr")

        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        semilagrangian_advection_markerchain!(
            chain, RungeKutta2(), @velocity(stokes), grid_vxi, xvi, dt,
        )
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
        inject_particles_phase!(particles, pPhases, (), ())
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        step += 1
        t += dt
    end

    return return_fields ? (;
            cost,
            history,
            phase_gradients = map(gradient -> Array(gradient.center), gradients),
            density_gradient = Array(stokes_ad.ρ),
            viscosity_gradient = Array(stokes_ad.viscosity.η),
            viscosity = Array(stokes.viscosity.η),
            rock_ratio = Array(ϕ.center),
        ) : nothing
end

if !@isdefined(NO_AUTORUN)
    n = 64
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    rayleigh_taylor2D_adjoint(igg, n, n)
end
