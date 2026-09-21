using Enzyme

"""
    enzyme_compute_PH_residual_V_sensitivity!(stokes, adjoint, ρg, dρg, _di, ni)

Differentiate the momentum residual with respect to pressure, stress, and the
two buoyancy-force arrays. The residual seeds are read from `adjoint.R`;
sensitivities accumulate in `adjoint.P`, `adjoint.τ`, and `dρg`.
"""
function enzyme_compute_PH_residual_V_sensitivity!(stokes, adjoint, ρg, dρg, _di, ni)
    @parallel (@idx ni) configcall = compute_PH_residual_V!(
        stokes.R.Rx,
        stokes.R.Ry,
        stokes.P,
        stokes.ΔPψ,
        stokes.τ.xx,
        stokes.τ.yy,
        stokes.τ.xy,
        ρg...,
        _di.center,
        _di.vertex,
    ) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        compute_PH_residual_V!,
        Enzyme.DuplicatedNoNeed(stokes.R.Rx, adjoint.R.Rx),
        Enzyme.DuplicatedNoNeed(stokes.R.Ry, adjoint.R.Ry),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.DuplicatedNoNeed(stokes.τ.xx, adjoint.τ.xx),
        Enzyme.DuplicatedNoNeed(stokes.τ.yy, adjoint.τ.yy),
        Enzyme.DuplicatedNoNeed(stokes.τ.xy, adjoint.τ.xy),
        Enzyme.DuplicatedNoNeed(ρg[1], dρg[1]),
        Enzyme.DuplicatedNoNeed(ρg[2], dρg[2]),
        Enzyme.Const(_di.center),
        Enzyme.Const(_di.vertex),
    )
    return nothing
end

# Pick the thermal-expansion accessor the density model actually provides: the
# melt-dependent models take the melt fraction, while the plain ones (PT_Density and
# friends) only define the single-argument method.
@inline model_thermal_expansion(model, ::NamedTuple{()}) = get_thermal_expansion(model)
@inline model_thermal_expansion(model, thermal_args::NamedTuple) =
    get_thermal_expansion(model, thermal_args)

# Differentiate the original GeoParams evaluations with local reverse seeds.
# Phase weighting is applied to the seeds by the caller, exactly once.
@inline function density_parameter_objective(model, args, ρ_seed, α_seed, thermal_args)
    result = ρ_seed * compute_density(model, args)
    if !iszero(α_seed)
        result += α_seed * model_thermal_expansion(model, thermal_args)
    end
    return result
end

@inline function enzyme_density_parameter_gradient(model, args, ρ_seed, α_seed, thermal_args)
    return Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        # `autodiff_deferred` takes the function as an annotation, unlike `autodiff`
        Enzyme.Const(density_parameter_objective),
        Enzyme.Active,
        Enzyme.Active(model),
        Enzyme.Const(args),
        Enzyme.Const(ρ_seed),
        Enzyme.Const(α_seed),
        Enzyme.Const(thermal_args),
    )[1][1]
end

"""
    enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt, controls = (;), gradients = nothing,
    )

Differentiate the constitutive kernel with respect to center and vertex
viscosity. Stress seeds come from `adjoint.τ`; viscosity sensitivities
accumulate in `adjoint.viscosity.η` and `adjoint.viscosity.ηv`.
When `gradients` is supplied, also accumulate derivatives with respect to the
selected multipliers. `compute_sensitivities!` converts these to material derivatives.
An empty `gradients` is treated like `nothing`: without a multiplier to differentiate,
`controls` is a constant, and seeding an empty `NamedTuple` as `Duplicated` would only
hand Enzyme an inactive shadow.
"""
function enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt, controls = (;), gradients = nothing
    )
    ni = size(phase_ratios.vertex)
    @parallel (@idx ni) configcall = compute_stress_DRYEL!(
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),
        (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
        stokes.τ.II,
        (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),
        (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy),
        stokes.EII_pl,
        stokes.ε_vol_pl,
        stokes.P,
        stokes.λ,
        stokes.λv,
        stokes.viscosity.η,
        stokes.viscosity.ηv,
        stokes.viscosity.η_vep,
        stokes.ΔPψ,
        rheology,
        phase_ratios.center,
        phase_ratios.vertex,
        λ_relaxation,
        dt,
        controls,
    ) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        compute_stress_DRYEL!,
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
            (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        ),
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
            (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        ),
        Enzyme.Const((stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c)),
        Enzyme.Const((stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy)),
        Enzyme.Const(stokes.τ.II),
        Enzyme.Const((stokes.ε.xx, stokes.ε.yy, stokes.ε.xy)),
        Enzyme.Const((stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)),
        Enzyme.Const(stokes.EII_pl),
        Enzyme.Const(stokes.ε_vol_pl),
        Enzyme.Const(stokes.P),
        Enzyme.Const(stokes.λ),
        Enzyme.Const(stokes.λv),
        Enzyme.DuplicatedNoNeed(stokes.viscosity.η, adjoint.viscosity.η),
        Enzyme.DuplicatedNoNeed(stokes.viscosity.ηv, adjoint.viscosity.ηv),
        Enzyme.Const(stokes.viscosity.η_vep),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios.center),
        Enzyme.Const(phase_ratios.vertex),
        Enzyme.Const(λ_relaxation),
        Enzyme.Const(dt),
        (isnothing(gradients) || isempty(gradients)) ?
            Enzyme.Const(controls) : Enzyme.Duplicated(controls, gradients),
    )
    return nothing
end
