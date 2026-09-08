using Test
using LinearAlgebra
using Random
using JustRelax, JustRelax.JustRelax2D
using CairoMakie: Axis, Figure, Label, axislegend, lines!, save, scatter!

# `NO_AUTORUN` suppresses the demo each example runs when it is loaded on its own, so that
# including them here only brings their functions into scope.
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
    include("ShearBand2D_adjoint.jl")
end

function evaluate_case(run_case, igg, perturbation; adjoint)
    Random.seed!(1234)
    return run_case(
        igg;
        viscosity_perturbation = perturbation,
        adjoint,
        return_fields = true,
        plot_results = false,
        solver_ϵ = 1.0e-8,
        verbose = false,
    )
end

function stationary_taylor_test(name, run_case, igg, ε)
    baseline = evaluate_case(run_case, igg, 0.0; adjoint = true)
    dJ = dot(baseline.viscosity_gradient, baseline.viscosity_direction) +
        dot(baseline.viscosity_gradient_vertex, baseline.viscosity_direction_vertex)
    scale = max(abs(baseline.cost), eps(Float64))
    remainder0 = similar(ε)
    remainder1 = similar(ε)

    for i in eachindex(ε)
        perturbed = evaluate_case(run_case, igg, ε[i]; adjoint = false)
        ΔJ = perturbed.cost - baseline.cost
        remainder0[i] = abs(ΔJ) / scale
        remainder1[i] = abs(ΔJ - ε[i] * dJ) / scale
    end

    slope0 = log(remainder0[3] / remainder0[1]) / log(ε[3] / ε[1])
    slope1 = log(remainder1[3] / remainder1[1]) / log(ε[3] / ε[1])
    @info "stationary Stokes viscosity Taylor test" example = name cost = baseline.cost dJ slope0 slope1 remainder0 remainder1
    return (; name, remainder0, remainder1, slope0, slope1)
end

function plot_taylor_tests(results, ε, filename)
    fig = Figure(; size = (1200, 450))
    Label(
        fig[0, 1:3],
        "|J(η + ε δη) − J(η) − ε ∇_ηJ ⋅ δη|";
        fontsize = 24,
    )
    for (column, result) in enumerate(results)
        axis = Axis(
            fig[1, column];
            title = result.name,
            xlabel = "viscosity perturbation ε",
            ylabel = column == 1 ? "normalized corrected remainder" : "",
            xscale = log10,
            yscale = log10,
        )
        # lines!(axis, ε, result.remainder1; label = "slope $(round(result.slope1; digits = 2))")
        scatter!(axis, ε, result.remainder1)
        lines!(axis, ε, result.remainder1[1] .* (ε ./ ε[1]) .^ 2; linestyle = :dot, label = "O(ε²)")
        axislegend(axis; position = :rb)
    end
    mkpath(dirname(filename))
    save(filename, fig)
    return fig
end

function main(igg; n = 16, figdir = joinpath("figures", "DYREL_adjoint_gradient_tests"))
    ε = 10.0 .^ range(-1, -6; length = 20)
    cases = (
        ("SinkingBlock2D", (igg; kwargs...) -> SinkingBlockCase.sinking_block2D(igg; nx = n, ny = n, ar = 1, kwargs...)),
        ("SinkingBlock2D VE", (igg; kwargs...) -> SinkingBlockVECase.sinking_block2D_VE(igg; nx = n, ny = n, ar = 1, nt = 0, kwargs...)),
        ("ShearBand2D", (igg; kwargs...) -> ShearBandCase.main(igg; nx = n, ny = n, nt = 1, kwargs...)),
    )
    results = map(cases) do (name, run_case)
        stationary_taylor_test(name, run_case, igg, ε)
    end

    filename = joinpath(figdir, "stationary_stokes_viscosity_taylor_tests.png")
    plot_taylor_tests(results, ε, filename)
    println("Saved stationary Stokes gradient plot to $filename")

    return results
end

n = 16

# `init_global_grid` throws if a global grid is already active, which is the case whenever this
# script is re-run in the same REPL session, or after one of the included examples was executed
# directly. Tear the stale grid down first -- with `finalize_MPI = false`, since MPI cannot be
# re-initialized in the same process -- so the grid is always rebuilt for the current `n`.
ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)

main(igg)
