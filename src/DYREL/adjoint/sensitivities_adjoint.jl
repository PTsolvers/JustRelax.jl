"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, ϕ::RockRatio, rheology, _di, ni, λ_relaxation, dt,
        igg, gradients, args; viscosity_cutoff = (-Inf, Inf), free_surface = false, air_phase = 0,
        constitutive_scratch = nothing, observation = nothing,
    )

Accumulate the sensitivities of the objective functional with respect to the material
parameters, given a converged adjoint state of [`solve_VariationalDYREL_adjoint!`](@ref) in
`stokes_ad.λV` and `stokes_ad.λP`.

Fills

  - `stokes_ad.ρ` -- sensitivity with respect to density,
  - `stokes_ad.viscosity.η` -- sensitivity with respect to the center viscosity, including the
    vertex stresses, which read its harmonic mean `harm_clamped(η)`; `stokes_ad.viscosity.ηv`
    stays zero because the variational stress kernel does not read `ηv`,
  - each `gradients` entry -- the accumulated phase-wise derivative with respect to every
    use of the requested parameter name. Center arrays include the transpose-interpolated
    vertex contribution; vertex arrays retain that contribution separately for diagnostics.

`gradients` comes from [`material_controls`](@ref) and is keyed directly by parameter name.
Each name is resolved in every supported material-function path before the sensitivity
kernels launch. Masked stress points are skipped. The viscosity and density parameter paths
use the same `air_phase` correction as the forward viscosity and buoyancy updates, and the
continuity-residual paths (bulk modulus and thermal expansion) carry the rock fraction of the
continuity residual.

The adjoint working arrays (`P`, `θ`, the stress tensor and the viscosity fields) are zeroed
on entry, so this has to be called *after* the adjoint iterations have converged and not in
between them. For a `:τII` objective, pass the resolved `observation` (from `observation_mask`):
the stress invariant also depends on the material directly, and that term is added here.
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
        observation = nothing,
    )
    periodic = periodic_dims(stokes)
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)
    # a stress objective depends on the material directly, not only through the solution
    !isnothing(observation) && observation.field === :τII && seed_observation!(observation, 1.0)

    # 1. Pull the masked momentum residual back to stress, pressure, plastic pressure
    #    correction, and buoyancy.
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, ϕ, _di, ni; free_surface_dt = dt * free_surface
    )

    # 2. Retain the direct seed for linear rheology, or form the implicit nonlinear seed.
    viscosity_seed = prepare_viscosity_parameter_seed!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, viscosity_cutoff; air_phase,
    )

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

    # 5. Differentiate buoyancy and the continuity residual, form the density derivative, and
    #    fold vertex material gradients into the center fields.
    finish_density_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic,
        _di; air_phase, ϕ,
    )

    return stokes_ad
end

# The variational stress kernel interpolates center viscosity to vertices. Its nonlinear
# pullback therefore returns one center-viscosity seed, including that interpolation path.
function prepare_viscosity_parameter_seed!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios,
        ϕ::JustRelax.RockRatio, λ_relaxation, dt, args, viscosity_cutoff;
        air_phase::Integer = 0,
    )
    isnothing(constitutive_scratch) && return stokes_ad.viscosity.η
    enzyme_prepare_variational_implicit_viscosity_τII!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, viscosity_cutoff; air_phase,
    )
    return constitutive_scratch.η
end

# Pull the converged stress seeds in `adjoint.τ` (and the plastic pressure correction seed
# `adjoint.θ`) back through the stress kernel, point by point with the rheology active. This
# fills `adjoint.viscosity.η` and adds the stress-path material-parameter derivatives to
# `gradients`. There is no vertex viscosity: the vertex contributions reach
# `adjoint.viscosity.η` through `harm_clamped`.
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
        map(entry -> entry.vertex, gradients), parameters, stress_args; periodic,
    )
    differentiate_pointwise_material_kernel!(
        compute_stress_DRYEL_center!, size(phases.center),
        map(entry -> entry.center, gradients), parameters, stress_args,
    )
    return nothing
end
