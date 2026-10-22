"""
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg,
        gradients = (;), args = (;); viscosity_cutoff = (-Inf, Inf),
    )

Accumulate the sensitivities of the objective functional with respect to the material
parameters, given a converged adjoint state in `stokes_ad.λV` and `stokes_ad.λP`.

Fills

  - `stokes_ad.ρ` -- sensitivity with respect to density,
  - `stokes_ad.viscosity.η` and `stokes_ad.viscosity.ηv` -- sensitivity with respect to
    viscosity at the centers and vertices.
  - each `gradients` entry -- the accumulated phase-wise derivative with respect to every
    use of the requested parameter name. Center arrays include the transpose-interpolated
    vertex contribution; vertex arrays retain that contribution separately for diagnostics.

`gradients` comes from [`material_controls`](@ref) and is keyed directly by parameter name.
Each name is resolved in every supported material-function path before the sensitivity
kernels launch.

The adjoint working arrays (`P`, `θ`, the stress tensor and the viscosity fields) are zeroed
on entry, so this has to be called *after* the adjoint iterations have converged and not in
between them.
"""
function compute_sensitivities!(
        stokes,
        stokes_ad,
        ρg,
        phase_ratios,
        rheology,
        _di,
        ni,
        λ_relaxation,
        dt,
        igg,
        gradients = (;),
        args = (;),
        ;
        viscosity_cutoff = (-Inf, Inf),
        free_surface = false,
        constitutive_scratch = nothing,
    )
    periodic = periodic_dims(stokes)
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)

    # 1. Pull the momentum residual back to stress, pressure, plastic pressure correction,
    #    and buoyancy.
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, _di, ni; free_surface_dt = dt * free_surface
    )

    # 2. Retain the direct seed for linear rheology, or form the implicit nonlinear seed.
    viscosity_seed = prepare_viscosity_parameter_seed!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios,
        λ_relaxation, dt, args, viscosity_cutoff,
    )

    # 3. Pull the corrected stress seeds back to strain rate, pressure, viscosity, and the
    #    material parameters used directly by the stress kernel.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, λ_relaxation, dt, periodic, gradients, args
    )

    # 4. Pull the saved viscosity seeds back to the material parameters of the viscosity law.
    compute_viscosity_parameter_gradients!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients;
        viscosity_seed,
    )

    # 5. Differentiate buoyancy and the continuity residual, form the density derivative, and
    #    fold vertex material gradients into the center fields.
    finish_density_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, args, dt, gradients, periodic,
        _di,
    )

    return stokes_ad
end

# The nonlinear seed follows the same local implicit constitutive pullback used during the
# adjoint iteration. Linear rheology has no feedback, so its direct viscosity seed is final.
function prepare_viscosity_parameter_seed!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios,
        λ_relaxation, dt, args, viscosity_cutoff,
    )
    isnothing(constitutive_scratch) &&
        return (stokes_ad.viscosity.η, stokes_ad.viscosity.ηv)
    enzyme_compute_implicit_viscosity_τII!(
        constitutive_scratch, stokes, stokes_ad, rheology, phase_ratios,
        λ_relaxation, dt, args, viscosity_cutoff,
    )
    return (constitutive_scratch.η, constitutive_scratch.ηv)
end

"""
    compute_stress_sensitivities!(
        stokes, adjoint, phases, rheology, λ_relaxation, dt, periodic, gradients
    )

Pull the converged stress seeds in `adjoint.τ` (and the plastic pressure correction seed
`adjoint.θ`) back through the separated stress kernel of the forward solve. Its vertex and
center functions are reverse-differentiated point by point with the rheology active. This fills the
viscosity sensitivities `adjoint.viscosity.η` / `ηv` and adds the stress-path material-parameter
derivatives to `gradients`. The τII viscosity refresh is left out, as in the linear-viscosity
adjoint.
"""
function compute_stress_sensitivities!(
        stokes, adjoint, phases, rheology, λ_relaxation, dt, periodic, gradients, args
    )
    names = keys(gradients)
    parameters = map(material -> _resolve_parameter_paths(material, names, _stress_parameter_paths), rheology)
    differentiate_pointwise_material_kernel!(
        compute_stress_DRYEL_vertex!,
        size(phases.vertex),
        map(entry -> entry.vertex, gradients),
        parameters,
        (
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy) => (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy) => nothing,
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy) => (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
            (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy) => nothing,
            stokes.EII_pl => nothing,
            stokes.P => adjoint.P,
            fluid_pressure(args, stokes.P) => nothing,
            stokes.λv => nothing,
            stokes.viscosity.ηv => adjoint.viscosity.ηv,
            rheology => Enzyme.Active,
            phases.center => nothing,
            phases.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            periodic => nothing,
        ),
    )
    differentiate_pointwise_material_kernel!(
        compute_stress_DRYEL_center!,
        size(phases.center),
        map(entry -> entry.center, gradients),
        parameters,
        (
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c) => (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c) => nothing,
            stokes.τ.II => adjoint.τ.II,
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy) => (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
            (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy) => nothing,
            stokes.EII_pl => nothing,
            stokes.ε_vol_pl => nothing,
            stokes.P => adjoint.P,
            fluid_pressure(args, stokes.P) => nothing,
            stokes.λ => nothing,
            stokes.viscosity.η => adjoint.viscosity.η,
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => adjoint.θ,
            rheology => Enzyme.Active,
            phases.center => nothing,
            λ_relaxation => nothing,
            dt => nothing,
        ),
    )
    return nothing
end
