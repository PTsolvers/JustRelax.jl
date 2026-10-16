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
        constitutive_scratch = nothing,
    )
    periodic = periodic_dims(stokes)
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)

    # 1. Pull the masked momentum residual back to stress, pressure, plastic pressure
    #    correction, and buoyancy.
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, ϕ, _di, ni; free_surface_dt = dt * free_surface
    )

    # 2. Use the direct center-viscosity seed for linear rheology. For nonlinear rheology,
    #    replace it with the seed obtained after eliminating the local stress-viscosity
    #    feedback; this already includes the harmonic vertex-viscosity contribution.
    viscosity_seed = if isnothing(constitutive_scratch)
        (stokes_ad.viscosity.η, stokes_ad.viscosity.ηv)
    else
        enzyme_prepare_variational_implicit_viscosity_τII!(
            constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios, ϕ,
            λ_relaxation, dt, args, viscosity_cutoff; air_phase,
        )
        (constitutive_scratch.η, stokes_ad.viscosity.ηv)
    end

    # 3. Pull the corrected stress seeds back to strain rate, pressure, center viscosity, and
    #    the material parameters used directly by the variational stress kernel.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, ϕ, rheology, λ_relaxation, dt, periodic, gradients, args
    )

    # 4. Pull the saved center-viscosity seed back to the material parameters of the
    #    viscosity law.
    compute_viscosity_parameter_gradients!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients;
        air_phase, viscosity_seed,
    )

    # 5. Differentiate buoyancy and thermal expansion, form the density derivative, and fold
    #    vertex material gradients into the center fields.
    finish_density_sensitivities!(
        stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic; air_phase, ϕ
    )

    return stokes_ad
end

# Masked counterpart of `compute_stress_sensitivities!`, through the variational stress kernel.
# It has no vertex viscosity: the vertex contributions reach `adjoint.viscosity.η` through
# `harm_clamped`, and `adjoint.viscosity.ηv` stays zero.
function compute_stress_sensitivities!(
        stokes, adjoint, phases, ϕ::JustRelax.RockRatio, rheology, λ_relaxation, dt, periodic,
        gradients, args,
    )
    names = keys(gradients)
    parameters = map(material -> _resolve_parameter_paths(material, names, _stress_parameter_paths), rheology)
    stress_args = (
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c) => (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy) => (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c) => nothing,
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy) => nothing,
        stokes.τ.II => adjoint.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy) => (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy) => nothing,
        stokes.EII_pl => nothing,
        stokes.ε_vol_pl => nothing,
        stokes.P => adjoint.P,
        fluid_pressure(args, stokes.P) => nothing,
        stokes.λ => nothing,
        stokes.λv => nothing,
        stokes.viscosity.η => adjoint.viscosity.η,
        stokes.viscosity.η_vep => nothing,
        stokes.ΔPψ => adjoint.θ,
        ϕ => nothing,
        rheology => Enzyme.Active,
        phases.center => nothing,
        phases.vertex => nothing,
        λ_relaxation => nothing,
        dt => nothing,
        periodic => nothing,
    )
    differentiate_pointwise_material_kernel!(
        compute_stress_DRYEL_vertex!, size(phases.vertex),
        map(entry -> entry.vertex, gradients), parameters, stress_args,
    )
    differentiate_pointwise_material_kernel!(
        compute_stress_DRYEL_center!, size(phases.center),
        map(entry -> entry.center, gradients), parameters, stress_args,
    )
    return nothing
end
