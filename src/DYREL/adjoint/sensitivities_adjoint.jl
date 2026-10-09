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
    )
    periodic = periodic_dims(stokes)
    prepare_sensitivities!(stokes_ad, rheology, ni, gradients, igg)

    # differntiates momentum equation w.r.t. stress, pressure and plastic pressuure correction
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, _di, ni; free_surface_dt = dt * free_surface
    )

    # Pull the stress seeds back through the stress kernel to the viscosity fields and the
    # stress-path material parameters.
    compute_stress_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, λ_relaxation, dt, periodic, gradients, args
    )

    compute_linear_viscosity_parameter_sensitivities!(
        stokes, stokes_ad, phase_ratios, rheology, args, viscosity_cutoff, gradients
    )

    finish_density_sensitivities!(stokes_ad, phase_ratios, rheology, args, dt, gradients, periodic)

    return stokes_ad
end

"""
    compute_stress_sensitivities!(
        stokes, adjoint, phases, rheology, λ_relaxation, dt, periodic, gradients
    )

Pull the converged stress seeds in `adjoint.τ` (and the plastic pressure correction seed
`adjoint.θ`) back through the fused stress kernel of the forward solve: its vertex and center
halves are reverse-differentiated point by point with the rheology active. This fills the
viscosity sensitivities `adjoint.viscosity.η` / `ηv` and adds the stress-path material-parameter
derivatives to `gradients`. The τII viscosity refresh is left out, as in the linear-viscosity
adjoint.
"""
function compute_stress_sensitivities!(
        stokes, adjoint, phases, rheology, λ_relaxation, dt, periodic, gradients, args
    )
    names = keys(gradients)
    parameters = map(material -> _resolve_parameter_paths(material, names, _stress_parameter_paths), rheology)
    # the forward kernel also writes the pressure correction θc = γ_eff·RP + ΔPψ; it does not feed
    # back into the stress, so scratch arrays keep the solver's own θc untouched
    θc = similar(stokes.P)
    γ_eff = zero(stokes.P)
    enzyme_stress_sensitivities!(
        compute_stress_viscosity_DRYEL_vertex!,
        compute_stress_viscosity_DRYEL_center!,
        size(phases.vertex),
        map(entry -> entry.center, gradients),
        map(entry -> entry.vertex, gradients),
        parameters,
        (
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
            stokes.viscosity.ηv => adjoint.viscosity.ηv,
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => adjoint.θ,
            θc => nothing,
            stokes.R.RP => nothing,
            γ_eff => nothing,
            rheology => Enzyme.Active,
            phases.center => nothing,
            phases.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            1.0 => nothing,
            (;) => nothing,
            (-Inf, Inf) => nothing,
            true => nothing,
            periodic => nothing,
        ),
    )
    return nothing
end
