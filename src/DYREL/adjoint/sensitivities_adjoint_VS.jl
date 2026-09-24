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
        stokes, stokes_ad, phase_ratios, rheology, λ_relaxation, dt, periodic, gradients; ϕ
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
