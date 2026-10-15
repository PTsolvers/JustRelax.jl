using Enzyme

"""
    enzyme_compute_∇V_strain_rate_RP!(
        stokes, adjoint, dyrel, rheology, phase_ratios, _di, ni, dt, args;
        do_strain_rate=true,
    )

`adjoint.ε` and `adjoint.R.RP` seed the strain-rate and pressure-residual outputs. The
resulting reverse-mode derivatives accumulate in `adjoint.V` and `adjoint.P`; all remaining
inputs are constant.
"""
function enzyme_compute_∇V_strain_rate_RP!(
        stokes,
        adjoint,
        dyrel,
        rheology,
        phase_ratios,
        _di,
        ni,
        dt,
        args;
        do_strain_rate = true,
    )
    ΔT = haskey(args, :ΔT) ? args.ΔT : nothing
    melt_fraction = haskey(args, :melt_fraction) ? args.melt_fraction : nothing

    enzyme_reverse_pointwise!(
        compute_∇V_strain_rate_RP_point!,
        ni .+ 1,
        (
            stokes.ε.xx => adjoint.ε.xx,
            stokes.ε.yy => adjoint.ε.yy,
            stokes.ε.xy => adjoint.ε.xy,
            stokes.V.Vx => adjoint.V.Vx,
            stokes.V.Vy => adjoint.V.Vy,
            stokes.R.RP => adjoint.R.RP,
            stokes.P => adjoint.P,
            stokes.P0 => nothing,
            stokes.Q => nothing,
            dyrel.ηb => nothing,
            _di.vertex => nothing,
            _di.velocity[1] => nothing,
            _di.velocity[2] => nothing,
            rheology => nothing,
            phase_ratios.center => nothing,
            ΔT => nothing,
            melt_fraction => nothing,
            dt => nothing,
            do_strain_rate => nothing,
        ),
    )
    return nothing
end

"""
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, _di, ni)

Differentiate the two-dimensional momentum-residual kernel.
`adjoint.R.Rx` and `adjoint.R.Ry` seed the outputs. The resulting
reverse-mode derivatives accumulate in `adjoint.V`, `adjoint.P`, `adjoint.τ`,
`adjoint.dρgx`, and `adjoint.ρ`.

Only this momentum-residual function is differentiated with Enzyme. The transpose of the
Schur-complement correction is applied separately by `update_adjoint_V_damping_DR!`.
enzyme_compute_PH_residual_V! is also used in the gradient calculation after the adjoint solve.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, _di, ni; free_surface_dt = 0
    )
    enzyme_reverse_pointwise!(
        compute_PH_residual_V_point!,
        ni,
        (
            stokes.R.Rx => adjoint.R.Rx,
            stokes.R.Ry => adjoint.R.Ry,
            stokes.V.Vx => adjoint.V.Vx,
            stokes.V.Vy => adjoint.V.Vy,
            stokes.P => adjoint.P,
            stokes.ΔPψ => adjoint.θ,
            stokes.τ.xx => adjoint.τ.xx,
            stokes.τ.yy => adjoint.τ.yy,
            stokes.τ.xy => adjoint.τ.xy,
            ρg[1] => adjoint.dρgx,
            ρg[2] => adjoint.ρ,
            _di.center => nothing,
            _di.vertex => nothing,
            free_surface_dt => nothing,
        ),
    )
    return nothing
end

"""
    enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, _di, ni, rheology, phases, args
    )

`adjoint.R.Rx` and `adjoint.R.Ry` seed the momentum-residual outputs. The resulting
reverse-mode derivatives accumulate in `adjoint.V`, `adjoint.P`, `adjoint.τ`,
`adjoint.dρgx`, and `adjoint.ρ`; the buoyancy derivative is additionally propagated through
the pressure-dependent density into `adjoint.P`. The adjoint solver requires
`args.P === stokes.P`.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, _di, ni, rheology, phases, args; free_surface_dt = 0
    )
    zero_buoyancy_adjoint!(adjoint)
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, _di, ni; free_surface_dt)
    enzyme_compute_ρg!(ρg, adjoint, phases, rheology, args, ni)
    return nothing
end

# The buoyancy adjoints collect the residual's ρg seeds, so they start from zero on every pass.
zero_buoyancy_adjoint!(adjoint) = foreach(A -> fill!(A, 0.0), (adjoint.dρgx, adjoint.ρ))

"""
    enzyme_compute_ρg!(ρg, adjoint, phases, rheology, args, ni; air_phase=0)

Differentiate the same pointwise buoyancy function used by the forward solver.
`adjoint.dρgx` and `adjoint.ρ` seed the horizontal and vertical buoyancy outputs through
`DuplicatedNoNeed`. The resulting pressure derivative accumulates in `adjoint.P`.
"""
function enzyme_compute_ρg!(
        ρg, adjoint, phases, rheology, args, ni; air_phase::Integer = 0
    )
    args_without_P, P = split_pressure_arg(args)
    isnothing(P) && return nothing
    enzyme_reverse_pointwise!(
        compute_ρg_kernel_point!,
        ni,
        (
            ρg => (adjoint.dρgx, adjoint.ρ),
            phases.center => nothing,
            rheology => nothing,
            args_without_P => nothing,
            P => adjoint.P,
            air_phase => nothing,
            ni => nothing,
        ),
    )
    return nothing
end

"""
    enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt, args,
        viscosity_cutoff, linear_viscosity, viscosity_target,
    )

Differentiate the separate two-dimensional stress kernel used by the normal
DYREL solver. For nonlinear viscosity, first correct the momentum stress seed
for the converged constitutive equations

```
τ = stress(ε, P, η)
η = viscosity(τII(τ)).
```

The correction uses the unrelaxed stress-based viscosity function. The corrected stress seeds
in `adjoint.τ`, together with the plastic-pressure-correction seed in `adjoint.θ`, are then
pulled back through `compute_stress_DRYEL_point!`. The resulting reverse-mode derivatives
accumulate in `adjoint.ε`, `adjoint.P`, `adjoint.viscosity.η`, and
`adjoint.viscosity.ηv`; neither the forward viscosity relaxation nor its iteration history is
differentiated.
"""
function enzyme_compute_stress_DRYEL!(
        stokes,
        adjoint,
        rheology,
        phase_ratios,
        λ_relaxation,
        dt,
        args,
        viscosity_cutoff,
        linear_viscosity,
        viscosity_target,
    )
    periodic = periodic_dims(stokes)

    # The viscosity shadows are scratch for one constitutive pullback. They must not carry the
    # relaxed forward iteration into the next adjoint iteration: the nonlinear correction below
    # differentiates the final, unrelaxed constitutive equations directly.
    fill!(adjoint.viscosity.η, 0)
    fill!(adjoint.viscosity.ηv, 0)

    if !linear_viscosity
        @parallel (@idx size(phase_ratios.vertex)) implicit_τII_viscosity_seed!(
            adjoint.viscosity.η,
            adjoint.viscosity.ηv,
            (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
            (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c),
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),
            stokes.P,
            fluid_pressure(args, stokes.P),
            stokes.λ,
            stokes.λv,
            stokes.viscosity.η,
            stokes.viscosity.ηv,
            stokes.EII_pl,
            rheology,
            phase_ratios.center,
            phase_ratios.vertex,
            λ_relaxation,
            dt,
            args,
            viscosity_cutoff,
            periodic,
        )
        enzyme_compute_viscosity_τII!(
            viscosity_target, stokes, adjoint, phase_ratios, args, rheology,
            viscosity_cutoff,
        )
    end

    enzyme_reverse_pointwise!(
        compute_stress_DRYEL_point!,
        size(phase_ratios.vertex),
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
            rheology => nothing,
            phase_ratios.center => nothing,
            phase_ratios.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            periodic => nothing,
        ),
    )
    return nothing
end

# Reverse the two pointwise viscosity evaluations used by the normal 2D solver. The local
# fixed-point correction above supplies the center and vertex viscosity seeds; Enzyme applies
# (∂η/∂τ)' to them with the exact forward viscosity kernel and without numerical relaxation.
function enzyme_compute_viscosity_point!(
        η_target, η_adjoint, ratios, tensor, tensor_adjoint, args, rheology, cutoff,
        fn_viscosity, fn_args; air_phase::Integer = 0,
    )
    enzyme_reverse_pointwise!(
        compute_viscosity_kernel_point!,
        size(η_target),
        (
            η_target => η_adjoint,
            one(eltype(η_target)) => nothing,
            ratios => nothing,
            tensor[1] => tensor_adjoint[1],
            tensor[2] => tensor_adjoint[2],
            tensor[3] => tensor_adjoint[3],
            args => nothing,
            rheology => nothing,
            air_phase => nothing,
            cutoff => nothing,
            fn_viscosity => nothing,
            fn_args => nothing,
        ),
    )
    return nothing
end

function enzyme_compute_viscosity_τII!(
        viscosity_target, stokes, adjoint, phase_ratios, args, rheology, cutoff;
        air_phase::Integer = 0,
    )
    η_target, ηv_target = viscosity_target
    enzyme_compute_viscosity_point!(
        η_target, adjoint.viscosity.η, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args;
        air_phase,
    )
    enzyme_compute_viscosity_point!(
        ηv_target, adjoint.viscosity.ηv, phase_ratios.vertex,
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
        (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args_vertex;
        air_phase,
    )
    return nothing
end

# Return the three stress components produced by `compute_local_stress` while exposing only the
# stored viscosity as the first argument. Forward-mode AD with a unit viscosity tangent therefore
# gives ∂τ/∂η without forming the complete constitutive Jacobian.
@inline function local_stress_from_viscosity(
        η, ε, τ_o, P, λ, λ_relaxation, rheology, ratio, dt, EII, Pf
    )
    result = compute_local_stress(
        ε, τ_o, η, P, λ, λ_relaxation, rheology, ratio, dt, EII, Pf
    )
    return result[1], result[2], result[3]
end

@inline function viscosity_fixed_point_map(
        η, ε, τ_o, P, λ, λ_relaxation, rheology, ratio, dt, EII, Pf,
        viscosity_args, cutoff, air_phase,
    )
    τ = local_stress_from_viscosity(
        η, ε, τ_o, P, λ, λ_relaxation, rheology, ratio, dt, EII, Pf
    )
    return compute_viscosity_value(
        η, τ, viscosity_phase_ratio(air_phase, ratio), viscosity_args, rheology, one(η), cutoff,
        compute_viscosity_τII,
    )
end

# Correct one stress seed for the implicit final-state relationship
#
#     τ = stress(ε, P, η),       η = viscosity(τII(τ)).
#
# For a stress seed `τ_bar`, the scalar local viscosity adjoint is
#
#     η_bar = dot(∂τ/∂η, τ_bar) / (1 - ∂viscosity(stress(η))/∂η).
#
# `enzyme_compute_viscosity_τII!` then applies `(∂η/∂τ)' * η_bar` with the exact pointwise
# viscosity kernel. No viscosity adjoint is retained between global λV/λP iterations.
@inline function implicit_τII_viscosity_adjoint(
        τ_bar, η, ε, τ_o, P, λ, λ_relaxation, rheology, ratio, dt, EII, Pf,
        viscosity_args, cutoff, air_phase = 0, η_seed = zero(η),
    )
    dτdη = only(
        Enzyme.autodiff_deferred(
            Enzyme.Forward,
            Enzyme.Const(local_stress_from_viscosity),
            Enzyme.Duplicated,
            Enzyme.Duplicated(η, one(η)),
            Enzyme.Const(ε),
            Enzyme.Const(τ_o),
            Enzyme.Const(P),
            Enzyme.Const(λ),
            Enzyme.Const(λ_relaxation),
            Enzyme.Const(rheology),
            Enzyme.Const(ratio),
            Enzyme.Const(dt),
            Enzyme.Const(EII),
            Enzyme.Const(Pf),
        ),
    )
    feedback = only(
        Enzyme.autodiff_deferred(
            Enzyme.Forward,
            Enzyme.Const(viscosity_fixed_point_map),
            Enzyme.Duplicated,
            Enzyme.Duplicated(η, one(η)),
            Enzyme.Const(ε),
            Enzyme.Const(τ_o),
            Enzyme.Const(P),
            Enzyme.Const(λ),
            Enzyme.Const(λ_relaxation),
            Enzyme.Const(rheology),
            Enzyme.Const(ratio),
            Enzyme.Const(dt),
            Enzyme.Const(EII),
            Enzyme.Const(Pf),
            Enzyme.Const(viscosity_args),
            Enzyme.Const(cutoff),
            Enzyme.Const(air_phase),
        ),
    )

    η_bar = (η_seed + sum(dτdη .* τ_bar)) / (one(feedback) - feedback)
    return η_bar
end

@parallel_indices (i, j) function implicit_τII_viscosity_seed!(
        η_bar,
        ηv_bar,
        τ_bar,
        τ_bar_v,
        τ_o,
        τ_ov,
        ε,
        P,
        Pf,
        λ,
        λv,
        η,
        ηv,
        EII_pl,
        rheology,
        phase_ratios_center,
        phase_ratios_vertex,
        λ_relaxation,
        dt,
        viscosity_args,
        cutoff,
        periodic,
    )
    ni = size(phase_ratios_center)
    @inbounds begin
        vertex = stress_DRYEL_vertex_state(
            τ_ov, ε, EII_pl, P, Pf, λv, ηv, phase_ratios_center,
            phase_ratios_vertex, periodic, i, j,
        )
        τbar_ij_v = τ_bar_v[1][i, j], τ_bar_v[2][i, j], τ_bar_v[3][i, j]
        ηv_bar[i, j] = implicit_τII_viscosity_adjoint(
            τbar_ij_v, vertex.η, vertex.ε, vertex.τ_o, vertex.P, vertex.λ,
            λ_relaxation, rheology, vertex.ratio, dt, vertex.EII, vertex.Pf,
            local_viscosity_args_vertex(viscosity_args, i, j), cutoff,
        )

        if i ≤ ni[1] && j ≤ ni[2]
            center = stress_DRYEL_center_state(
                τ_o, ε, EII_pl, P, Pf, λ, η, phase_ratios_center, i, j
            )
            τbar_ij = τ_bar[1][i, j], τ_bar[2][i, j], τ_bar[3][i, j]
            η_bar[i, j] = implicit_τII_viscosity_adjoint(
                τbar_ij, center.η, center.ε, center.τ_o, center.P, center.λ,
                λ_relaxation, rheology, center.ratio, dt, center.EII, center.Pf,
                local_viscosity_args(viscosity_args, i, j), cutoff,
            )
        end
    end
    return nothing
end
