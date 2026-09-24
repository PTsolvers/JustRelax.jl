"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, ϕ::RockRatio, rheology, _di, ni, λ_relaxation, dt,
        igg, gradients, args; viscosity_cutoff = (-Inf, Inf), free_surface = false, air_phase = 0,
    )

Variational counterpart of the non-variational `compute_sensitivities!`, given a converged
adjoint state of [`solve_VariationalDYREL_adjoint!`](@ref). Fills the same fields, with these
differences:

  - masked stress points are skipped, and the vertex stress uses `harm_clamped(η)` as in the
    forward kernel;
  - `stokes_ad.viscosity.η` holds the full center-viscosity sensitivity, including the vertex
    contributions through the harmonic mean, and `stokes_ad.viscosity.ηv` stays zero because
    the variational stress kernel does not read `ηv`;
  - the viscosity and density parameter paths use the same `air_phase` correction as the
    forward viscosity and buoyancy updates, and the thermal-expansion path carries the rock
    fraction of the continuity residual.
"""
function compute_sensitivities!(
        stokes,
        stokes_ad,
        ρg,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        rheology,
        _di,
        ni,
        λ_relaxation,
        dt,
        igg,
        gradients,
        args;
        viscosity_cutoff = (-Inf, Inf),
        free_surface = false,
        air_phase::Integer = 0,
    )
    periodic = periodic_dims(stokes)
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)

    # differentiates the masked momentum equation w.r.t. stress, pressure, plastic pressure
    # correction and buoyancy
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, ϕ, _di, ni; free_surface_dt = dt * free_surface
    )

    # Pull back the local stress update to its material parameters. This only reads the stress
    # seeds, so it has to run before the reverse stress kernel below consumes them.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, ϕ, rheology, λ_relaxation, dt, periodic, gradients
    )

    # viscosity sensitivity: the reverse stress kernel folds the harmonic vertex viscosity back
    # into the centers
    enzyme_compute_stress_DRYEL!(stokes, stokes_ad, rheology, phase_ratios, ϕ, λ_relaxation, dt)

    compute_linear_viscosity_parameter_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients; air_phase
    )

    finish_density_sensitivities!(
        stokes_ad, phase_ratios, rheology, args, dt, gradients, periodic; air_phase, ϕ
    )

    return stokes_ad
end

# Masked counterpart of `compute_stress_sensitivities!`: material parameters only, the viscosity
# sensitivity comes from the reverse variational stress kernel.
function compute_stress_sensitivities!(
        stokes, adjoint, phases, ϕ::JustRelax.RockRatio, rheology, λ_relaxation, dt, periodic,
        gradients,
    )
    names = keys(gradients)
    centers = map(entry -> entry.center, gradients)
    vertices = map(entry -> entry.vertex, gradients)
    for (p, material) in enumerate(rheology)
        parameters = _resolve_parameter_paths(material, names, _stress_parameter_paths)

        @parallel (@idx size(phases.vertex)) stress_sensitivity_kernel!(
            centers, vertices, parameters,
            material, p, phases.center, phases.vertex,
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),
            stokes.EII_pl, stokes.P, stokes.λ, stokes.λv, stokes.viscosity.η,
            (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
            (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
            adjoint.θ, λ_relaxation, dt, periodic, ϕ,
        )
    end
    return nothing
end
