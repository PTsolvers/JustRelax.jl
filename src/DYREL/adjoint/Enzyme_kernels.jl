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
        viscosity_cutoff, linear_viscosity, constitutive_scratch,
    )

Differentiate the separate two-dimensional stress kernel used by the normal DYREL solver.
The incoming stress seeds in `adjoint.τ` come from the momentum-residual pullback. For linear
viscosity they are pulled back through `compute_stress_DRYEL_point!` directly:

```
τ_bar = (∂momentum_residual/∂τ)' residual_seed
ε_bar += (∂τ/∂ε)' τ_bar
P_bar += (∂τ/∂P)' τ_bar.
```

For nonlinear viscosity, the converged local state satisfies

```
τ = stress(ε, P, η)
η = viscosity(τII(τ)).
```

The nonlinear reverse sweep has three steps:

1. Reverse the stress kernel to obtain its direct viscosity seed.
2. Reverse the viscosity and stress kernels with unit viscosity seeds to obtain the scalar
   fixed-point feedback, then divide the direct seed by `1 - feedback`.
3. Reverse viscosity with the implicit seed and stress with the corrected stress seeds.

The viscosity adjoint arrays are temporary storage for these steps. The forward viscosity
relaxation and its iteration history are not differentiated.
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
        constitutive_scratch,
    )
    # These are local constitutive scratch fields, not additional adjoint unknowns. Clear values
    # left by the preceding global λV/λP iteration before forming the current implicit pullback.
    fill!(adjoint.viscosity.η, 0)
    fill!(adjoint.viscosity.ηv, 0)

    if !linear_viscosity
        enzyme_compute_implicit_viscosity_τII!(
            constitutive_scratch, stokes, adjoint, rheology, phase_ratios,
            λ_relaxation, dt, args, viscosity_cutoff,
        )
    end

    # Step 3: pull the corrected stress seeds through τ = stress(ε, P, η). This produces the
    # strain-rate and pressure terms used by the λV/λP adjoint iteration.
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, λ_relaxation, dt, args;
        τ_bar = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        τv_bar = (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        τII_bar = adjoint.τ.II,
        ε_bar = (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        P_bar = adjoint.P,
        η_bar = adjoint.viscosity.η,
        ηv_bar = adjoint.viscosity.ηv,
        θ_bar = adjoint.θ,
    )
    return nothing
end

function enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, λ_relaxation, dt, args;
        τ_bar = nothing, τv_bar = nothing, τII_bar = nothing, ε_bar = nothing,
        P_bar = nothing, η_bar = nothing, ηv_bar = nothing, θ_bar = nothing,
    )
    enzyme_reverse_pointwise!(
        compute_stress_DRYEL_point!,
        size(phase_ratios.vertex),
        (
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c) => τ_bar,
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy) => τv_bar,
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c) => nothing,
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy) => nothing,
            stokes.τ.II => τII_bar,
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy) => ε_bar,
            (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy) => nothing,
            stokes.EII_pl => nothing,
            stokes.ε_vol_pl => nothing,
            stokes.P => P_bar,
            fluid_pressure(args, stokes.P) => nothing,
            stokes.λ => nothing,
            stokes.λv => nothing,
            stokes.viscosity.η => η_bar,
            stokes.viscosity.ηv => ηv_bar,
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => θ_bar,
            rheology => nothing,
            phase_ratios.center => nothing,
            phase_ratios.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            periodic_dims(stokes) => nothing,
        ),
    )
    return nothing
end

# Reverse one pointwise viscosity evaluation. `η_adjoint` seeds its viscosity output and Enzyme
# adds `(∂η/∂τ)'η̄` to `tensor_adjoint`. `η_target` is a copy of the converged forward viscosity,
# so differentiating this scratch output does not overwrite the forward state.
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
        viscosity_target, viscosity_seed, stress_seed, stokes, phase_ratios, args,
        rheology, cutoff;
        air_phase::Integer = 0,
    )
    η_target, ηv_target = viscosity_target
    η_bar, ηv_bar = viscosity_seed
    τ_bar, τv_bar = stress_seed
    enzyme_compute_viscosity_point!(
        η_target, η_bar, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        τ_bar,
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args;
        air_phase,
    )
    enzyme_compute_viscosity_point!(
        ηv_target, ηv_bar, phase_ratios.vertex,
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
        τv_bar,
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args_vertex;
        air_phase,
    )
    return nothing
end

zero_constitutive_stress_seed!(scratch) = foreach(
    A -> fill!(A, 0), (scratch.τ..., scratch.τ_v..., scratch.τII, scratch.θ)
)

function enzyme_compute_viscosity_feedback_τII!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    # Differentiate η(τ(η)) with unit viscosity seeds. These local scalar feedback fields are
    # used by both the implicit constitutive pullback and the adjoint DYREL parameter estimate.
    zero_constitutive_stress_seed!(scratch)
    fill!(scratch.η, 1)
    fill!(scratch.ηv, 1)
    enzyme_compute_viscosity_τII!(
        (scratch.η_target, scratch.ηv_target), (scratch.η, scratch.ηv),
        (scratch.τ, scratch.τ_v), stokes, phase_ratios, args, rheology, cutoff,
    )
    fill!(scratch.η, 0)
    fill!(scratch.ηv, 0)
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, λ_relaxation, dt, args;
        τ_bar = scratch.τ, τv_bar = scratch.τ_v, τII_bar = scratch.τII,
        η_bar = scratch.η, ηv_bar = scratch.ηv,
    )
    copyto!(scratch.feedback, scratch.η)
    copyto!(scratch.feedback_v, scratch.ηv)
    scratch.feedback_λ[] = λ_relaxation
    return nothing
end

# The forward state is fixed throughout an adjoint solve. Cache the local viscosity feedback
# for each stress-relaxation value, which avoids two Enzyme sweeps on every DR iteration.
function ensure_constitutive_viscosity_feedback!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    if !isequal(scratch.feedback_λ[], λ_relaxation)
        enzyme_compute_viscosity_feedback_τII!(
            scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
        )
    end
    return nothing
end

function compute_constitutive_tangent_amplification!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    ensure_constitutive_viscosity_feedback!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    feedback_max = max(maximum_mpi(scratch.feedback), maximum_mpi(scratch.feedback_v))
    isfinite(feedback_max) && feedback_max < 1 || error(
        "The nonlinear constitutive tangent is singular or unstable: maximum feedback = $feedback_max"
    )
    unit_amplification = one(eltype(scratch.η))
    @. scratch.η = max(unit_amplification, abs(inv(1 - scratch.feedback)))
    @. scratch.ηv = max(unit_amplification, abs(inv(1 - scratch.feedback_v)))
    return nothing
end

function enzyme_compute_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    τ_bar = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c)
    τv_bar = (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy)

    # Direct path: (∂stress/∂η)' * stress_seed.
    foreach(copyto!, scratch.τ, τ_bar)
    foreach(copyto!, scratch.τ_v, τv_bar)
    copyto!(scratch.τII, adjoint.τ.II)
    copyto!(scratch.θ, adjoint.θ)
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, λ_relaxation, dt, args;
        τ_bar = scratch.τ, τv_bar = scratch.τ_v, τII_bar = scratch.τII,
        η_bar = adjoint.viscosity.η, ηv_bar = adjoint.viscosity.ηv,
        θ_bar = scratch.θ,
    )

    # only caclulate feedbck variables if they are not already calculated before
    ensure_constitutive_viscosity_feedback!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    # Apply the local scalar implicit-function correction. The feedback is a 1×1 quantity,
    # so it equals its transpose.
    @. adjoint.viscosity.η /= 1 - scratch.feedback
    @. adjoint.viscosity.ηv /= 1 - scratch.feedback_v

    # Preserve the implicit viscosity seeds for the later material-parameter pullback, then add
    # `(∂η/∂τ)'η_bar` to the original stress seeds.
    copyto!(scratch.η, adjoint.viscosity.η)
    copyto!(scratch.ηv, adjoint.viscosity.ηv)
    enzyme_compute_viscosity_τII!(
        (scratch.η_target, scratch.ηv_target),
        (adjoint.viscosity.η, adjoint.viscosity.ηv), (τ_bar, τv_bar),
        stokes, phase_ratios, args, rheology, cutoff,
    )
    fill!(adjoint.viscosity.η, 0)
    fill!(adjoint.viscosity.ηv, 0)
    return nothing
end
