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

"""
    random_direction(rng, ni)

Draw a random, unit-norm viscosity perturbation direction `δ = (; center, vertex)`.

The centre and vertex fields are normalised jointly, so `‖δ‖₂ = 1` over the whole staggered
pair rather than over each grid separately.
"""
function random_direction(rng, ni)
    center = randn(rng, ni...)
    vertex = randn(rng, (ni .+ 1)...)
    nrm = sqrt(sum(abs2, center) + sum(abs2, vertex))
    return (; center = center ./ nrm, vertex = vertex ./ nrm)
end

"""
    smooth_direction(ni)

Sample one smooth continuous perturbation on the center and vertex grids. The positive
mean reduces cancellation of the directional derivative while the sinusoidal part still tests
spatially varying viscosity.
"""
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

"""
    evaluate_case(run_case, igg, ε, δ; adjoint)

Run one forward (and optionally adjoint) solve with the viscosity scaled cell-wise by
`exp(ε δ)`.

Parameterising the perturbation multiplicatively keeps the viscosity positive for any `ε`
and makes the chain-rule direction `dη/dε = η ⊙ δ`, which is what the adjoint gradient has to
be contracted against.
"""
function evaluate_case(run_case, igg, ε, δ; adjoint)
    Random.seed!(1234)
    η_multiplier = (; center = exp.(ε .* δ.center), vertex = exp.(ε .* δ.vertex))
    return run_case(
        igg;
        η_multiplier,
        adjoint,
        return_fields = true,
        plot_results = adjoint,
        # The Taylor test resolves differences in J far below the cost itself, so the forward
        # and adjoint solves have to be converged well past the usual 1e-6. At 1e-8 the pure
        # viscous SinkingBlock case -- the stiffest of the three, having no elastic
        # regularisation -- still carries a ~5% gradient error purely from under-convergence.
        # Do not tighten much further: below ~1e-11 the two forward solves stop agreeing to
        # the precision the finite difference needs, and the reference becomes solver noise.
        solver_ϵ = 1.0e-10,
        verbose = false,
    )
end

function loglog_slope(ε, remainder; fit_range = (1.0e-4, 1.0e-1))
    indices = findall(x -> fit_range[1] ≤ x ≤ fit_range[2], ε)
    length(indices) ≥ 2 || throw(ArgumentError("the slope fit needs at least two ε values"))
    x = log10.(ε[indices])
    y = log10.(remainder[indices])
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    return dot(x .- x̄, y .- ȳ) / sum(abs2, x .- x̄)
end

function stationary_taylor_test(name, run_case, igg, ε, δ; fit_range = (1.0e-4, 1.0e-1))
    baseline = evaluate_case(run_case, igg, 0.0, δ; adjoint = true)
    # dJ/dε = Σ (∂J/∂η) (dη/dε), and dη/dε = η ⊙ δ for the multiplicative parameterisation
    dJ_adjoint = dot(baseline.viscosity_gradient, baseline.viscosity .* δ.center) +
        dot(baseline.viscosity_gradient_vertex, baseline.viscosity_vertex .* δ.vertex)
    scale = max(abs(baseline.cost), eps(Float64))
    remainder0 = similar(ε)
    remainder1 = similar(ε)

    for i in eachindex(ε)
        perturbed = evaluate_case(run_case, igg, ε[i], δ; adjoint = false)
        ΔJ = perturbed.cost - baseline.cost
        remainder0[i] = abs(ΔJ) / scale
        remainder1[i] = abs(ΔJ - ε[i] * dJ_adjoint) / scale
    end

    slope0 = loglog_slope(ε, remainder0; fit_range)
    slope1 = loglog_slope(ε, remainder1; fit_range)
    @info "stationary Stokes viscosity Taylor test" example = name cost = baseline.cost dJ_adjoint fit_range slope0 slope1 remainder0 remainder1
    return (; name, remainder0, remainder1, fit_range, slope0, slope1)
end

function plot_taylor_tests(results, ε, filename)
    nplots = length(results)
    fig = Figure(; size = (400 * nplots, 450))
    Label(
        fig[0, 1:nplots],
        "Taylor test of adjoint viscosity gradient", ;
        fontsize = 22,
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
        lines!(axis, ε, result.remainder1; label = "fitted slope $(round(result.slope1; digits = 2))")
        scatter!(axis, ε, result.remainder1)
        reference_index = findfirst(x -> result.fit_range[1] ≤ x ≤ result.fit_range[2], ε)
        lines!(axis, ε, result.remainder1[reference_index] .* (ε ./ ε[reference_index]) .^ 2; linestyle = :dot, label = "O(ε²)")
        axislegend(axis; position = :rb)
    end
    mkpath(dirname(filename))
    save(filename, fig)
    return fig
end

function main(igg; n = 16, figdir = joinpath("figures", "DYREL_adjoint_gradient_tests"))
    ε = 10.0 .^ range(-3, -8; length = 10)
    cases = (
        ("SinkingBlock2D", (igg; kwargs...) -> SinkingBlockCase.sinking_block2D(igg; nx = n, ny = n, ar = 1, figdir = joinpath(figdir, "SinkingBlock2D"), kwargs...)),
        ("SinkingBlock2D VE", (igg; kwargs...) -> SinkingBlockVECase.sinking_block2D_VE(igg; nx = n, ny = n, ar = 1, nt = 4, figdir = joinpath(figdir, "SinkingBlock2D_VE"), kwargs...)),
        ("ShearBand2D", (igg; kwargs...) -> ShearBandCase.main(igg; nx = n, ny = n, nt = 15, figdir = joinpath(figdir, "ShearBand2D"), kwargs...)),
    )
    # One physical perturbation field, sampled consistently at centers and vertices and
    # shared by every case and every ε.
    δ = smooth_direction((n, n))
    results = map(cases) do (name, run_case)
        fit_range = name == "ShearBand2D" ? (1.0e-4, 1.0e-3) : (1.0e-4, 1.0e-1)
        stationary_taylor_test(name, run_case, igg, ε, δ; fit_range)
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
