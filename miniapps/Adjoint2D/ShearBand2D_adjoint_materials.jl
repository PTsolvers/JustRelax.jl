# One stationary shear-band problem for verifying all non-complex-viscosity material
# gradients. The forward rheology uses density, linear viscosity, elasticity and
# Drucker-Prager plasticity. All corresponding material gradients are evaluated and plotted.
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using JustPIC
using GeoParams
using CairoMakie: Axis, Colorbar, DataAspect, Figure, Label, Relative, heatmap!, hidexdecorations!, hideydecorations!, rowsize!, save

@init_parallel_stencil(Threads, Float64, 2)

const backend    = JustRelax.CPUBackend
const JP_backend = JustPIC.CPU
const SHEARBAND_PHASE_NAMES          = ("Matrix", "Inclusion")

@parallel_indices (i, j) function init_material_shear_band_phases!(
        phases, xc, yc, center, radius
    )
    inside = (xc[i] - center[1])^2 + (yc[j] - center[2])^2 ≤ radius^2
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

material_shear_band_parameters() = (
    (;
        ρ0 = 1.0, α = 0.04, β = 0.02, T0 = 0.2, P0 = 0.1,
        η = 1.0, G = 1.0, Kb = 5.0,
        C = 0.30, ϕ = 30.0, Ψ = 1.0, η_vp = 1.0e-2,
    ),
    (;
        ρ0 = 1.15, α = 0.07, β = 0.04, T0 = 0.1, P0 = 0.2,
        η = 0.5, G = 0.5, Kb = 3.0,
        C = 0.30, ϕ = 30.0, Ψ = 1.0, η_vp = 1.0e-2,
    ),
)

function material_shear_band_rheology(;
        parameters = material_shear_band_parameters()
    )
    return ntuple(2) do p
        values = parameters[p]
        density = PT_Density(;
            values.ρ0, values.α, values.β, values.T0, values.P0
        )
        viscosity = LinearViscous(; values.η)
        elasticity = ConstantElasticity(; values.G, values.Kb)
        plasticity = DruckerPrager_regularised(;
            values.C, values.ϕ, values.Ψ, values.η_vp
        )
        SetMaterialParams(;
            Name = SHEARBAND_PHASE_NAMES[p],
            Phase = p,
            Density = density,
            Gravity = ConstantGravity(; g = 0.02),
            CompositeRheology = CompositeRheology((
                viscosity, elasticity, plasticity,
            )),
            Elasticity = elasticity,
            Plasticity = plasticity,
        )
    end
end

function material_shear_band_gradient_fields(gradients)
    return map(
        gradient -> dropdims(sum(gradient.center; dims = 1); dims = 1), gradients
    )
end

function material_shear_band_colorrange(A)
    magnitude = maximum(abs, A)
    return iszero(magnitude) ? (-1.0, 1.0) : (-magnitude, magnitude)
end

function material_shear_band_plot_array(field)
    scalar(value) = value isa Number ? value : only(value)
    return [scalar(field[i, j]) for i in axes(field, 1), j in axes(field, 2)]
end

function plot_material_shear_band_gradients(figdir, step, grid, gradients, parameter_names)
    fields = material_shear_band_gradient_fields(gradients)
    parameters_per_row = 2
    nrows = cld(length(parameter_names), parameters_per_row)
    figure = Figure(size = (1400, 2100))
    npanels_per_row = parameters_per_row
    Label(
        figure[0, 1:(2npanels_per_row)],
        "Spatial material gradients (Matrix + Inclusion)";
        fontsize = 24,
    )

    for (n, name) in enumerate(parameter_names)
        row = fld(n - 1, parameters_per_row) + 1
        panel = mod(n - 1, parameters_per_row) + 1
        column = 2panel - 1
        field = fields[name]
        axis = Axis(
            figure[row, column];
            aspect = DataAspect(), title = "∂J/∂$name", xlabel = "x", ylabel = "y",
        )
        values = material_shear_band_plot_array(field)
        plot = heatmap!(
            axis,
            grid.xci...,
            values;
            colormap = :vik,
            colorrange = material_shear_band_colorrange(values),
        )
        Colorbar(figure[row, column + 1], plot)
        row < nrows && hidexdecorations!(axis; grid = false)
        panel > 1 && hideydecorations!(axis; grid = false)
    end
    for row in 1:nrows
        rowsize!(figure.layout, row, Relative(1 / nrows))
    end
    save(
        joinpath(figdir, "gradients", "all_phases_step$(lpad(step, 3, '0')).png"),
        figure,
    )
    return nothing
end

function plot_material_shear_band_state(figdir, step, grid, stokes, phase_ratios)
    ni = size(stokes.P)
    Vxv = @zeros(ni .+ 1...)
    Vyv = @zeros(ni .+ 1...)
    velocity2vertex!(Vxv, Vyv, @velocity(stokes)...)
    inclusion = [
        phase_ratios.center[i, j][2] for
            i in axes(phase_ratios.center, 1), j in axes(phase_ratios.center, 2)
    ]
    fields = (
        ("inclusion fraction", grid.xci, inclusion, false),
        ("Vx", grid.xvi, Vxv, true),
        ("Vy", grid.xvi, Vyv, true),
        ("P", grid.xci, stokes.P, true),
        ("η effective", grid.xci, stokes.viscosity.η, false),
        ("τII", grid.xci, stokes.τ.II, false),
    )
    figure = Figure(size = (1700, 700))
    for (n, (title, coordinates, field, diverging)) in enumerate(fields)
        row = fld(n - 1, 3) + 1
        column = mod(n - 1, 3) + 1
        axis = Axis(
            figure[row, 2column - 1];
            aspect = DataAspect(), title, xlabel = "x", ylabel = "y",
        )
        values = material_shear_band_plot_array(field)
        plot = if diverging
            heatmap!(
                axis,
                coordinates...,
                values;
                colormap = :vik,
                colorrange = material_shear_band_colorrange(values),
            )
        else
            heatmap!(axis, coordinates..., values; colormap = :batlow)
        end
        Colorbar(figure[row, 2column], plot)
        row == 1 && hidexdecorations!(axis; grid = false)
        column > 1 && hideydecorations!(axis; grid = false)
    end
    save(joinpath(figdir, "forward", "step$(lpad(step, 3, '0')).png"), figure)
    return nothing
end

function shear_band2D_adjoint_materials(
        igg;
        nx = 32,
        ny = nx,
        nt = 15,
        figdir = "ShearBand2D_adjoint_materials",
        adjoint = true,
        plot_results = true,
        return_fields = false,
        solver_ϵ = 1.0e-6,
        verbose = true,
        rheology = material_shear_band_rheology(),
        final_rheology = nothing,
        η_multiplier = nothing,
    )
    ni = nx, ny
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    (; xci, xvi) = grid

    phase_ratios = PhaseRatios(JP_backend, length(rheology), ni)
    inclusion_center = (0.5, 0.5)
    inclusion_radius = 0.12
    @parallel (@idx ni) init_material_shear_band_phases!(
        phase_ratios.center, xci..., inclusion_center, inclusion_radius
    )
    @parallel (@idx ni .+ 1) init_material_shear_band_phases!(
        phase_ratios.vertex, xvi..., inclusion_center, inclusion_radius
    )

    #### Stokes arrays ####
    stokes = StokesArrays(backend, ni)

    #### Adjoint arrays ####
    # Select fields for which you want to calculate sensitivities
    target_parameters = (
        :ρ0, :α, :β, :T0, :P0, :η, :G, :Kb, :C, :ϕ, :Ψ, :η_vp,
    )
    stokes_ad = AdjointStokesArrays(backend, ni)
    gradients = material_controls(
        backend,
        ni,
        target_parameters;
        nphases = length(rheology),
    )

    dt = 1 / 6
    temperature = PTArray(backend)([
        0.8 + 0.2 * x + 0.1 * y for x in range(-grid.di.center[1] / 2, 1 + grid.di.center[1] / 2; length = nx + 2),
        y in range(-grid.di.center[2] / 2, 1 + grid.di.center[2] / 2; length = ny + 2)
    ])
    ΔT = @fill(0.02, ni .+ 2...)
    args = (; T = temperature, P = stokes.P, dt, ΔT)
    ρg = @zeros(ni...), @zeros(ni...)
    viscosity_cutoff = (-Inf, Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    strain_rate = 1.0
    stokes.V.Vx .= PTArray(backend)([
        x * strain_rate for x in xvi[1], _ in 1:(ny + 2)
    ])
    stokes.V.Vy .= PTArray(backend)([
        -y * strain_rate for _ in 1:(nx + 2), y in xvi[2]
    ])
    @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0
    @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    observation = (;
        field = :Vy,
        center = (inclusion_center[1], inclusion_center[2] + inclusion_radius / 2),
        half_width = (
            max(inclusion_radius, grid.di.center[1]),
            max(inclusion_radius / 2, grid.di.center[2]),
        ),
    )
    if plot_results
        take(joinpath(figdir, "forward"))
        take(joinpath(figdir, "gradients"))
    end
    dyrel = DYREL(
        backend,
        stokes,
        rheology,
        phase_ratios,
        grid.di,
        dt;
        ϵ = solver_ϵ,
    )

    for step in 1:nt
        run_adjoint = adjoint && step == nt
        step_rheology = step == nt && !isnothing(final_rheology) ? final_rheology : rheology
        step_η_multiplier = step == nt ? η_multiplier : nothing
        solve_DYREL!(
            stokes,
            stokes_ad,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            step_rheology,
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
            viscosity_cutoff,
            linear_viscosity = true,
            η_multiplier = step_η_multiplier,
            adjoint = run_adjoint,
            observation,
            gradients = run_adjoint ? gradients : (;),
        )
        tensor_invariant!(stokes.τ)
        if plot_results && step in (1, nt)
            plot_material_shear_band_state(figdir, step, grid, stokes, phase_ratios)
        end
        if plot_results && run_adjoint
            plot_material_shear_band_gradients(
                figdir, step, grid, gradients, target_parameters
            )
        end
    end

    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    cost = sum(@view stokes.V.Vy[mask.i, mask.j])
    combined_gradients = material_shear_band_gradient_fields(gradients)
    phase_gradients = map(gradient -> Array(gradient.center), gradients)
    material_gradients = (;
        ρ = Array(stokes_ad.ρ),
        η_effective = Array(stokes_ad.viscosity.η),
        map(
            name -> name => Array(combined_gradients[name]),
            target_parameters,
        )...,
    )
    return return_fields ? (;
        cost,
        gradients = material_gradients,
        phase_gradients,
        viscosity_gradient = Array(stokes_ad.viscosity.η),
        viscosity_gradient_vertex = Array(stokes_ad.viscosity.ηv),
        viscosity = Array(stokes.viscosity.η),
        viscosity_vertex = Array(stokes.viscosity.ηv),
        yielded_center = Array(stokes.λ .> 0),
        yielded_vertex = Array(stokes.λv .> 0),
        target_parameters,
    ) : nothing
end

if !@isdefined(NO_AUTORUN)
    n = 1
    nx = 32*n
    ny = 32*n
    ImplicitGlobalGrid.grid_is_initialized() &&
        finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    @time shear_band2D_adjoint_materials(igg; nx, ny)
end
