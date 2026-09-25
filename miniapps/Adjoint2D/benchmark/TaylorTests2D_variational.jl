# Taylor tests of the variational DYREL adjoint (`solve_VariationalDYREL!`): the cases of
# TaylorTests2D.jl (sinking block, visco-elastic sinking block, plastic shear band), each run
# through the variational solver with rock everywhere (ϕ = 1, `variational = true`). The remainder
# |J(m + εδm) − J(m) − ε∇J⋅δm| of a correct gradient falls as O(ε²).
#
# As in TaylorTests2D.jl, every perturbation acts on the last time step only, the step the
# adjoint differentiates.
using JustRelax, JustRelax.JustRelax2D

# helpers and cases of the plain suite: smooth_direction, loglog_slope, viscosity_taylor_test,
# perturb_material_parameters, plot_taylor_tests, SinkingBlockCase, SinkingBlockVECase,
# ShearBandCase
module PlainTaylorTests
    const NO_AUTORUN = true
    include("TaylorTests2D.jl")
end
const PT = PlainTaylorTests

# Material-parameter Taylor test on the variational shear band. The plain suite's version runs
# the plain solver, so this one passes `variational = true`; otherwise it is the same test.
function material_taylor_test(name, igg, ε; n, nt, direction = (1.0, 1.0), fit_range)
    SB = PT.ShearBandCase
    parameters = SB.material_shear_band_parameters()
    rheology = SB.material_shear_band_rheology(; parameters)
    run_case(; kwargs...) = SB.shear_band2D_adjoint_materials(
        igg;
        nx = n,
        ny = n,
        nt,
        rheology,
        variational = true,
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
        perturbed_parameters = PT.perturb_material_parameters(parameters, name, step, direction)
        final_rheology = SB.material_shear_band_rheology(; parameters = perturbed_parameters)
        perturbed = run_case(; adjoint = false, final_rheology)
        abs(perturbed.cost - baseline.cost - step * directional_derivative) / scale
    end
    slope = PT.loglog_slope(ε, remainder, fit_range)
    @info "material Taylor test (variational)" parameter = name slope directional_derivative
    return (; name = "ShearBand2D: $name", parameter = name, remainder, slope, fit_range)
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
    direction = PT.smooth_direction((n, n))
    viscous_case(igg; kwargs...) = PT.SinkingBlockCase.sinking_block2D(
        igg;
        nx = n, ny = n, ar = 1, plot_results = false, variational = true,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )
    viscoelastic_case(igg; kwargs...) = PT.SinkingBlockVECase.sinking_block2D_VE(
        igg;
        nx = n, ny = n, ar = 1, nt = ve_steps, plot_results = false, variational = true,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )
    shear_viscosity_case(igg; kwargs...) =
        PT.ShearBandCase.shear_band2D_adjoint_materials(
        igg;
        nx = n, ny = n, nt = shear_steps, plot_results = false, variational = true,
        solver_ϵ = 1.0e-10, verbose = false, kwargs...,
    )

    results = Any[
        PT.viscosity_taylor_test(
            "SinkingBlock2D", viscous_case, igg, ε, direction;
            fit_range = (1.0e-4, 1.0e-2),
        ),
        PT.viscosity_taylor_test(
            "SinkingBlock2D VE", viscoelastic_case, igg, ε, direction;
            fit_range = (1.0e-4, 1.0e-2),
        ),
        PT.viscosity_taylor_test(
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

    filename = joinpath(figdir, "taylor_tests_variational.png")
    PT.plot_taylor_tests(results, ε, filename)
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
