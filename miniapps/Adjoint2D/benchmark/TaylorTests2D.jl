using LinearAlgebra
using Random
using JustRelax, JustRelax.JustRelax2D
using CairoMakie: Axis, Figure, Label, axislegend, lines!, save, scatter!

module SinkingBlockCase
    const NO_AUTORUN = true
    include("SinkingBlock2D_adjoint.jl")
end

module SinkingBlockVECase
    const NO_AUTORUN = true
    include("SinkingBlock2D_VE_adjoint.jl")
end

module ShearBandCase
    const NO_AUTORUN = true
    include("ShearBand2D_adjoint_materials.jl")
end

function smooth_direction(ni)
    nx, ny = ni
    center = [
        1.0 + 0.25 * sinpi(2 * (i - 0.5) / nx) * cospi(2 * (j - 0.5) / ny) for
            i in 1:nx, j in 1:ny
    ]
    vertex = [
        1.0 + 0.25 * sinpi(2 * (i - 1) / nx) * cospi(2 * (j - 1) / ny) for
            i in 1:(nx + 1), j in 1:(ny + 1)
    ]
    amplitude = max(maximum(abs, center), maximum(abs, vertex))
    return (; center = center ./ amplitude, vertex = vertex ./ amplitude)
end

function loglog_slope(ε, remainder, fit_range)
    indices = findall(eachindex(ε)) do i
        fit_range[1] ≤ ε[i] ≤ fit_range[2] && remainder[i] > 0
    end
    length(indices) ≥ 2 || throw(ArgumentError("the slope fit needs at least two points"))
    x = log10.(ε[indices])
    y = log10.(remainder[indices])
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return dot(x .- x̄, y .- ȳ) / sum(abs2, x .- x̄)
end

function viscosity_taylor_test(name, run_case, igg, ε, direction; fit_range)
    function evaluate(step; adjoint)
        Random.seed!(1234)
        multiplier = (;
            center = exp.(step .* direction.center),
            vertex = exp.(step .* direction.vertex),
        )
        return run_case(igg; η_multiplier = multiplier, adjoint, return_fields = true)
    end

    baseline = evaluate(0.0; adjoint = true)
    directional_derivative =
        dot(baseline.viscosity_gradient, baseline.viscosity .* direction.center) +
        dot(
        baseline.viscosity_gradient_vertex,
        baseline.viscosity_vertex .* direction.vertex,
    )
    scale = max(abs(baseline.cost), eps(Float64))
    remainder = map(ε) do step
        perturbed = evaluate(step; adjoint = false)
        abs(perturbed.cost - baseline.cost - step * directional_derivative) / scale
    end
    slope = loglog_slope(ε, remainder, fit_range)
    @info "viscosity Taylor test" example = name slope directional_derivative
    return (; name, parameter = :η, remainder, slope, fit_range)
end

function perturb_material_parameters(parameters, name, step, direction)
    return ntuple(length(parameters)) do phase
        value = parameters[phase][name]
        perturbation = NamedTuple{(name,)}((value * (1 + step * direction[phase]),))
        merge(parameters[phase], perturbation)
    end
end

function material_taylor_test(name, igg, ε; n, nt, direction = (1.0, 1.0), fit_range)
    parameters = ShearBandCase.material_shear_band_parameters()
    rheology = ShearBandCase.material_shear_band_rheology(; parameters)
    run_case(; kwargs...) = ShearBandCase.shear_band2D_adjoint_materials(
        igg;
        nx = n,
        ny = n,
        nt,
        rheology,
        return_fields = true,
        plot_results = false,
        solver_ϵ = 1.0e-10,
        verbose = false,
        kwargs...,
    )

    baseline = run_case(; adjoint = true)
    gradient = baseline.phase_gradients[name]
    directional_derivative = sum(eachindex(parameters)) do phase
        parameters[phase][name] * direction[phase] * sum(selectdim(gradient, 1, phase))
    end
    scale = max(abs(baseline.cost), eps(Float64))
    remainder = map(ε) do step
        perturbed_parameters = perturb_material_parameters(
            parameters, name, step, direction
        )
        final_rheology = ShearBandCase.material_shear_band_rheology(;
            parameters = perturbed_parameters
        )
        perturbed = run_case(; adjoint = false, final_rheology)
        abs(perturbed.cost - baseline.cost - step * directional_derivative) / scale
    end
    slope = loglog_slope(ε, remainder, fit_range)
    @info "material Taylor test" parameter = name slope directional_derivative
    return (; name = "ShearBand2D: $name", parameter = name, remainder, slope, fit_range)
end

function plot_taylor_tests(results, ε, filename)
    ncolumns = min(3, length(results))
    nrows = cld(length(results), ncolumns)
    figure = Figure(; size = (450ncolumns, 400nrows))
    Label(
        figure[0, 1:ncolumns],
        "|J(m + εδm) − J(m) − ε∇J⋅δm|";
        fontsize = 22,
    )
    for (index, result) in enumerate(results)
        row, column = fldmod1(index, ncolumns)
        axis = Axis(
            figure[row, column];
            title = result.name,
            xlabel = "perturbation ε",
            ylabel = column == 1 ? "normalized remainder" : "",
            xscale = log10,
            yscale = log10,
        )
        lines!(
            axis, ε, result.remainder;
            label = "slope $(round(result.slope; digits = 2))",
        )
        scatter!(axis, ε, result.remainder)
        reference = findfirst(step -> result.fit_range[1] ≤ step ≤ result.fit_range[2], ε)
        lines!(
            axis,
            ε,
            result.remainder[reference] .* (ε ./ ε[reference]) .^ 2;
            linestyle = :dot,
            label = "O(ε²)",
        )
        axislegend(axis; position = :rb)
    end
    mkpath(dirname(filename))
    save(filename, figure)
    return figure
end

function main(
        igg;
        n = 16,
        ve_steps = 3,
        shear_steps = 10,
        ε = 10.0 .^ (-3:-1:-4),
        material_parameters = (:G, :C, :ϕ, :Ψ, :η_vp),
        figdir = joinpath("figures", "Adjoint2D", "TaylorTests"),
    )
    direction = smooth_direction((n, n))
    viscous_case(igg; kwargs...) = SinkingBlockCase.sinking_block2D(
        igg;
        nx = n, ny = n, ar = 1, plot_results = false,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )
    viscoelastic_case(igg; kwargs...) = SinkingBlockVECase.sinking_block2D_VE(
        igg;
        nx = n, ny = n, ar = 1, nt = ve_steps, plot_results = false,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )
    shear_viscosity_case(igg; kwargs...) =
        ShearBandCase.shear_band2D_adjoint_materials(
        igg;
        nx = n, ny = n, nt = shear_steps, plot_results = false,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )

    results = Any[
        viscosity_taylor_test(
            "SinkingBlock2D", viscous_case, igg, ε, direction;
            fit_range = (1.0e-4, 1.0e-2),
        ),
        viscosity_taylor_test(
            "SinkingBlock2D VE", viscoelastic_case, igg, ε, direction;
            fit_range = (1.0e-4, 1.0e-2),
        ),
        viscosity_taylor_test(
            "ShearBand2D", shear_viscosity_case, igg, ε, direction;
            fit_range = (1.0e-4, 1.0e-3),
        ),
    ]
    append!(
        results,
        map(material_parameters) do parameter
            material_taylor_test(
                parameter, igg, ε;
                n, nt = shear_steps, fit_range = (1.0e-4, 1.0e-3),
            )
        end,
    )

    filename = joinpath(figdir, "taylor_tests.png")
    plot_taylor_tests(results, ε, filename)
    println("Saved Taylor-test plot to $filename")
    return results
end

if !@isdefined(NO_AUTORUN)
    n = 16
    ImplicitGlobalGrid.grid_is_initialized() &&
        finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    main(igg; n)
end
