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

    # Pull the stress seeds back through the masked stress kernel to the viscosity field and the
    # stress-path material parameters. The harmonic vertex viscosity is folded back into the
    # centers.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, ϕ, rheology, λ_relaxation, dt, periodic, gradients
    )

    compute_linear_viscosity_parameter_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients; air_phase
    )

    finish_density_sensitivities!(
        stokes_ad, phase_ratios, rheology, args, dt, gradients, periodic; air_phase, ϕ
    )

    return stokes_ad
end

# Masked counterpart of `compute_stress_sensitivities!`, through the variational stress kernel.
# It has no vertex viscosity: the vertex contributions reach `adjoint.viscosity.η` through
# `harm_clamped`, and `adjoint.viscosity.ηv` stays zero.
function compute_stress_sensitivities!(
        stokes, adjoint, phases, ϕ::JustRelax.RockRatio, rheology, λ_relaxation, dt, periodic,
        gradients,
    )
    names = keys(gradients)
    parameters = map(material -> _resolve_parameter_paths(material, names, _stress_parameter_paths), rheology)
    enzyme_stress_sensitivities!(
        compute_stress_DRYEL_vertex!,
        compute_stress_DRYEL_center!,
        size(phases.vertex),
        map(entry -> entry.center, gradients),
        map(entry -> entry.vertex, gradients),
        parameters,
        Val(17),
        Enzyme.DuplicatedNoNeed((stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c), (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c)),
        Enzyme.DuplicatedNoNeed((stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy), (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy)),
        Enzyme.Const((stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c)),
        Enzyme.Const((stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy)),
        Enzyme.DuplicatedNoNeed(stokes.τ.II, adjoint.τ.II),
        Enzyme.DuplicatedNoNeed((stokes.ε.xx, stokes.ε.yy, stokes.ε.xy), (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy)),
        Enzyme.Const((stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)),
        Enzyme.Const(stokes.EII_pl),
        Enzyme.Const(stokes.ε_vol_pl),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.Const(stokes.λ),
        Enzyme.Const(stokes.λv),
        Enzyme.DuplicatedNoNeed(stokes.viscosity.η, adjoint.viscosity.η),
        Enzyme.Const(stokes.viscosity.η_vep),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.Const(ϕ),
        Enzyme.Active(rheology),                                 # argument 17
        Enzyme.Const(phases.center),
        Enzyme.Const(phases.vertex),
        Enzyme.Const(λ_relaxation),
        Enzyme.Const(dt),
        Enzyme.Const(periodic),
    )
    return nothing
end
