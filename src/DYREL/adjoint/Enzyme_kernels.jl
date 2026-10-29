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
        stokes, adjoint, ρg, _di, ni, rheology, phases, args; update_material = true
    )

`adjoint.R.Rx` and `adjoint.R.Ry` seed the momentum-residual outputs. The resulting
reverse-mode derivatives accumulate in `adjoint.V`, `adjoint.P`, `adjoint.τ`,
`adjoint.dρgx`, and `adjoint.ρ`; the buoyancy derivative is additionally propagated through
the pressure-dependent density into `adjoint.P`, unless `update_material = false`, for which
the forward keeps ρg fixed. The adjoint solver requires `args.P === stokes.P`.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, _di, ni, rheology, phases, args; free_surface_dt = 0,
        update_material = true,
    )
    zero_buoyancy_adjoint!(adjoint)
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, _di, ni; free_surface_dt)
    update_material && enzyme_compute_ρg!(ρg, adjoint, phases, rheology, args, ni)
    return nothing
end

# The buoyancy adjoints collect the residual's ρg seeds, so they start from zero on every pass.
zero_buoyancy_adjoint!(adjoint) = foreach(A -> fill!(A, 0.0), (adjoint.dρgx, adjoint.ρ))

"""
    enzyme_compute_ρg!(ρg, adjoint, phases, rheology, args, ni; air_phase=0)

Differentiate the same pointwise buoyancy function used by the forward solver.
`adjoint.dρgx` and `adjoint.ρ` seed the horizontal and vertical buoyancy outputs through
`DuplicatedNoNeed`. The resulting pressure derivative accumulates in `adjoint.P`, which is
annotated `Duplicated` as a temporary Enzyme workaround (see `ReadOnlyShadow`).
"""
function enzyme_compute_ρg!(
        ρg, adjoint, phases, rheology, args, ni; air_phase::Integer = 0
    )
    # the forward does not update ρg for a constant density (`update_ρg!`), so ρg(P) has no derivative
    isconstant(rheology) isa ConstantDensityTrait && return nothing
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
            # Temporary: `Duplicated` instead of `DuplicatedNoNeed` (see `ReadOnlyShadow`).
            P => ReadOnlyShadow(adjoint.P),
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

Each location's viscosity feeds back only through its own stress, so the implicit pullback is
local: η̄ = (∂τ/∂η)ᵀ τ̄ / (1 − F) with F = (∂η/∂τ)(∂τ/∂η), then τ̄ += (∂η/∂τ)ᵀ η̄ and
P̄ += (∂η/∂P)ᵀ η̄. These local derivatives are fixed during an adjoint solve and computed once
(`implicit_viscosity_scratch`), so an iteration reverses only the stress kernel, with the
corrected stress seeds. The forward viscosity relaxation and its iteration history are not
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
        );
        periodic = periodic_dims(stokes),
    )
    return nothing
end

# Reverse one pointwise viscosity evaluation. `η_adjoint` seeds its viscosity output and Enzyme
# adds `(∂η/∂τ)'η̄` to `tensor_adjoint`, and `(∂η/∂P)'η̄` to `P_adjoint` when it is given.
# `η_target` is a copy of the converged forward viscosity, so differentiating this scratch
# output does not overwrite the forward state.
function enzyme_compute_viscosity_point!(
        η_target, η_adjoint, ratios, tensor, tensor_adjoint, args, rheology, cutoff,
        fn_viscosity, fn_args; air_phase::Integer = 0, P_adjoint = nothing,
    )
    if !isnothing(P_adjoint)
        # `args.P` goes in as an argument of its own, so that its shadow can be seeded
        enzyme_reverse_pointwise!(
            compute_viscosity_pressure_point!,
            size(η_target),
            (
                η_target => η_adjoint,
                one(eltype(η_target)) => nothing,
                ratios => nothing,
                tensor[1] => tensor_adjoint[1],
                tensor[2] => tensor_adjoint[2],
                tensor[3] => tensor_adjoint[3],
                args.P => P_adjoint,
                merge(args, (; P = nothing)) => nothing,
                rheology => nothing,
                air_phase => nothing,
                cutoff => nothing,
                fn_viscosity => nothing,
                fn_args => nothing,
            ),
        )
        return nothing
    end
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
        air_phase::Integer = 0, P_bar = nothing,
    )
    η_target, ηv_target = viscosity_target
    η_bar, ηv_bar = viscosity_seed
    τ_bar, τv_bar = stress_seed
    P_adjoint = viscosity_pressure_adjoint(args, stokes, P_bar)
    enzyme_compute_viscosity_point!(
        η_target, η_bar, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        τ_bar,
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args;
        air_phase, P_adjoint,
    )
    enzyme_compute_viscosity_point!(
        ηv_target, ηv_bar, phase_ratios.vertex,
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
        τv_bar,
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args_vertex;
        air_phase, P_adjoint,
    )
    return nothing
end

# The viscosity depends on the solved pressure only when `args.P` is `stokes.P`; any other
# pressure in `args` is a fixed input.
viscosity_pressure_adjoint(_, _, ::Nothing) = nothing
viscosity_pressure_adjoint(args, stokes, P_bar) =
    hasproperty(args, :P) && args.P === stokes.P ? P_bar : nothing

@inline function compute_viscosity_pressure_point!(
        η, ν, ratios, Axx, Ayy, Axyv, P, args, rheology, air_phase::Integer, cutoff,
        fn_viscosity::F1, fn_args::F2, I::Vararg{Integer, 2},
    ) where {F1, F2}
    compute_viscosity_kernel_point!(
        η, ν, ratios, Axx, Ayy, Axyv, merge(args, (; P)), rheology, air_phase, cutoff,
        fn_viscosity, fn_args, I...,
    )
    return nothing
end

"""
    implicit_viscosity_scratch(stokes)

Storage of the implicit viscosity pullback of the normal DYREL adjoint. The forward state does
not change during an adjoint solve, so the local derivatives are computed once per solve and
stress relaxation (`ensure_constitutive_tangent!`):

- `s`: ∂(τxx, τyy, τxy, τII, ΔPψ)/∂η at centers, `s_v`: ∂(τxx, τyy, τxy)/∂ηv at vertices;
- `v`, `v_v`: ∂η/∂(τxx, τyy, τxy); `vP`, `vP_v`: ∂η/∂P (at a vertex, of its averaged P);
- `feedback`, `feedback_v`: F = (∂η/∂τ)(∂τ/∂η).

`η` and `ηv` receive the implicit viscosity seeds, which the material gradients reuse.
"""
function implicit_viscosity_scratch(stokes)
    center() = similar(stokes.viscosity.η)
    vertex() = similar(stokes.viscosity.ηv)
    return (;
        η_target = copy(stokes.viscosity.η),
        ηv_target = copy(stokes.viscosity.ηv),
        η = center(),
        ηv = vertex(),
        feedback = center(),
        feedback_v = vertex(),
        feedback_λ = Ref{Any}(nothing),
        s = ntuple(_ -> center(), 5),
        s_v = ntuple(_ -> vertex(), 3),
        v = ntuple(_ -> center(), 3),
        v_v = ntuple(_ -> vertex(), 3),
        vP = center(),
        vP_v = vertex(),
        P_v = vertex(),
        P_v_seed = vertex(),
    )
end

# ∂η/∂τ and ∂η/∂P of one viscosity evaluation, by forward sweeps with unit tangents.
function viscosity_tangent!(
        v, vP, η_target, ratios, tensor, stokes, args, rheology, cutoff, fn_args;
        air_phase::Integer = 0,
    )
    unit, null = fill!(similar(tensor[1]), 1), fill!(similar(tensor[1]), 0)
    pressure = !isnothing(viscosity_pressure_adjoint(args, stokes, true))
    P_seed = fill!(similar(stokes.P), 0)
    for k in 1:4
        if k == 4
            fill!(vP, 0)
            pressure || break
            fill!(P_seed, 1)
        end
        tangent = k == 4 ? vP : v[k]
        # the point reads the previous viscosity, with a zero weight
        fill!(tangent, 0)
        seeds = ntuple(d -> d == k ? unit : null, 3)
        constant = (
            rheology => nothing, air_phase => nothing, cutoff => nothing,
            compute_viscosity_τII => nothing, fn_args => nothing,
        )
        point = (
            η_target => tangent, one(eltype(η_target)) => nothing, ratios => nothing,
            tensor[1] => seeds[1], tensor[2] => seeds[2], tensor[3] => seeds[3],
        )
        if pressure
            enzyme_forward_pointwise!(
                compute_viscosity_pressure_point!, size(η_target),
                (point..., args.P => P_seed, merge(args, (; P = nothing)) => nothing, constant...),
            )
        else
            enzyme_forward_pointwise!(
                compute_viscosity_kernel_point!, size(η_target),
                (point..., args => nothing, constant...),
            )
        end
    end
    return nothing
end

# F = (∂η/∂τ)(∂τ/∂η) through the stress components; τII and ΔPψ do not enter the viscosity.
function compute_viscosity_feedback!(F, v, s)
    @. F = v[1] * s[1] + v[2] * s[2] + v[3] * s[3]
    return nothing
end

function compute_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    # ∂τ/∂η: each stress reads only the viscosity at its own location
    fill!(scratch.η, 1)
    fill!(scratch.ηv, 1)
    enzyme_forward_pointwise!(
        compute_stress_DRYEL_point!,
        size(phase_ratios.vertex),
        (
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c) => scratch.s[1:3],
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy) => scratch.s_v,
            (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c) => nothing,
            (stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy) => nothing,
            stokes.τ.II => scratch.s[4],
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy) => nothing,
            (stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy) => nothing,
            stokes.EII_pl => nothing,
            stokes.ε_vol_pl => nothing,
            stokes.P => nothing,
            fluid_pressure(args, stokes.P) => nothing,
            stokes.λ => nothing,
            stokes.λv => nothing,
            stokes.viscosity.η => scratch.η,
            stokes.viscosity.ηv => scratch.ηv,
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => scratch.s[5],
            rheology => nothing,
            phase_ratios.center => nothing,
            phase_ratios.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            periodic_dims(stokes) => nothing,
        ),
    )
    viscosity_tangent!(
        scratch.v, scratch.vP, scratch.η_target, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c), stokes, args, rheology, cutoff,
        local_viscosity_args,
    )
    viscosity_tangent!(
        scratch.v_v, scratch.vP_v, scratch.ηv_target, phase_ratios.vertex,
        (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy), stokes, args, rheology, cutoff,
        local_viscosity_args_vertex,
    )
    compute_viscosity_feedback!(scratch.feedback, scratch.v, scratch.s)
    compute_viscosity_feedback!(scratch.feedback_v, scratch.v_v, scratch.s_v)
    scratch.feedback_λ[] = λ_relaxation
    return nothing
end

# The outer and inner iterations may use different stress relaxations; recompute only then.
function ensure_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    isequal(scratch.feedback_λ[], λ_relaxation) || compute_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    return nothing
end

function compute_constitutive_tangent_amplification!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    ensure_constitutive_tangent!(
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

# η̄ = (∂τ/∂η)ᵀ τ̄ / (1 − F), then τ̄ += (∂η/∂τ)ᵀ η̄ and P̄ += (∂η/∂P)ᵀ η̄, at one location.
# `direct` holds the part of (∂τ/∂η)ᵀ τ̄ from stresses at other locations, if any.
@parallel_indices (I...) function implicit_viscosity_seed!(
        η̄, direct, τ̄, s, v, P̄, vP, F,
    )
    @inbounds begin
        seed = initial_seed(direct, η̄, I...)
        for k in eachindex(s)
            seed += s[k][I...] * τ̄[k][I...]
        end
        seed /= 1 - F[I...]
        η̄[I...] = seed
        for k in eachindex(v)
            τ̄[k][I...] += v[k][I...] * seed
        end
        P̄[I...] += vP[I...] * seed
    end
    return nothing
end

@inline initial_seed(::Nothing, η̄, I::Vararg{Integer, N}) where {N} = zero(eltype(η̄))
@inline initial_seed(direct, η̄, I::Vararg{Integer, N}) where {N} = @inbounds direct[I...]

# The pressure that a vertex viscosity reads, averaged as by `local_viscosity_args_vertex`.
@inline function vertex_viscosity_pressure_point!(P_v, P, I::Vararg{Integer, 2})
    @inbounds P_v[I...] = local_viscosity_args_vertex((; P), I...).P
    return nothing
end

function enzyme_compute_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    ensure_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, λ_relaxation, dt, args, cutoff,
    )
    τ̄ = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c, adjoint.τ.II, adjoint.θ)
    τ̄_v = (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy)
    # a vertex viscosity reads the average pressure of its cells: seed it, then spread it
    fill!(scratch.P_v_seed, 0)
    @parallel (@idx size(scratch.η)) implicit_viscosity_seed!(
        scratch.η, nothing, τ̄, scratch.s, scratch.v, adjoint.P, scratch.vP,
        scratch.feedback,
    )
    @parallel (@idx size(scratch.ηv)) implicit_viscosity_seed!(
        scratch.ηv, nothing, τ̄_v, scratch.s_v, scratch.v_v, scratch.P_v_seed,
        scratch.vP_v, scratch.feedback_v,
    )
    enzyme_reverse_pointwise!(
        vertex_viscosity_pressure_point!,
        size(scratch.P_v),
        (scratch.P_v => scratch.P_v_seed, stokes.P => adjoint.P),
    )
    return nothing
end
