## ENZYME WRAPPERS FOR THE VARIATIONAL (RockRatio ϕ) DYREL KERNELS
#
# Reverse-mode counterparts of the masked kernels used by `_solve_VariationalDYREL!`
# (solver_VS.jl). They mirror the wrappers in Enzyme_kernels.jl and differ only in the extra
# `ϕ::JustRelax.RockRatio` argument, which is always `Enzyme.Const`: the rock ratio, the
# validity masks derived from it and the marker chain are frozen during a step's adjoint.
# Rows and cells the forward kernels mask out write constants, so their reverse seeds are
# dropped by the differentiation itself; masking of the adjoint unknowns is left to the caller.
# 2D only, like the variational DYREL solver.

"""
    enzyme_compute_∇V_strain_rate_RP!(
        stokes, adjoint, dyrel, rheology, phase_ratios, ϕ::RockRatio, _di, ni, dt, args;
        do_strain_rate=true,
    )

Masked counterpart of the non-variational wrapper. `adjoint.ε` and `adjoint.R.RP` seed the
strain-rate and pressure-residual outputs. The resulting reverse-mode derivatives accumulate
in `adjoint.V` and `adjoint.P`. The rock-fraction scaling of the continuity residual is
differentiated along with it.
"""
function enzyme_compute_∇V_strain_rate_RP!(
        stokes,
        adjoint,
        dyrel,
        rheology,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
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
            ϕ => nothing,
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
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, ϕ::RockRatio, _di, ni; free_surface_dt=0)

Differentiate the masked Powell–Hestenes momentum residual, including the ϕ-weighted
free-surface stabilization term (`free_surface_dt = dt * free_surface`; `0` disables it).
`adjoint.R.Rx` and `adjoint.R.Ry` seed the residual outputs. The resulting reverse-mode
derivatives accumulate in `adjoint.V`, `adjoint.P`, `adjoint.τ`, `adjoint.dρgx`, and
`adjoint.ρ`.

Only this momentum-residual function is differentiated with Enzyme. The transpose of the
Schur-complement correction is applied separately by `update_adjoint_V_damping_DR!`.
enzyme_compute_PH_residual_V! is also used in the gradient calculation after the adjoint solve.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, ϕ::JustRelax.RockRatio, _di, ni; free_surface_dt = 0
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
            ϕ => nothing,
            _di.center => nothing,
            _di.vertex => nothing,
            free_surface_dt => nothing,
        ),
    )
    return nothing
end

"""
    enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, ϕ::RockRatio, _di, ni, rheology, phases, args;
        free_surface_dt=0, air_phase=0,
    )

`adjoint.R.Rx` and `adjoint.R.Ry` seed the masked momentum-residual outputs. The resulting
reverse-mode derivatives accumulate in `adjoint.V`, `adjoint.P`, `adjoint.τ`,
`adjoint.dρgx`, and `adjoint.ρ`; the buoyancy derivative is additionally propagated through
the pressure-dependent density into `adjoint.P`. `air_phase` must match the forward
`update_ρg!` call, which drops that phase from the density average. The adjoint solver requires
`args.P === stokes.P`.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, ϕ::JustRelax.RockRatio, _di, ni, rheology, phases, args;
        free_surface_dt = 0, air_phase::Integer = 0,
    )
    zero_buoyancy_adjoint!(adjoint)
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, ϕ, _di, ni; free_surface_dt)
    enzyme_compute_ρg!(ρg, adjoint, phases, rheology, args, ni; air_phase)
    return nothing
end

"""
    enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ::RockRatio, λ_relaxation, dt, args,
    )

Differentiate the masked variational DYREL stress kernel. `adjoint.τ` and `adjoint.θ` seed the
stress and plastic-pressure-correction outputs; reverse-mode derivatives accumulate in
`adjoint.ε`, `adjoint.P`, and `adjoint.viscosity.η`.

Unlike the normal solver, the variational solver stores only center viscosity. A vertex stress
uses `harm_clamped(η)` of four surrounding centers. Reversing the stress kernel therefore maps
both center- and vertex-stress seeds into the center field `adjoint.viscosity.η`; there is no
independent vertex-viscosity adjoint.

For nonlinear viscosity, `enzyme_compute_variational_stress_DRYEL!` first forms the implicit
center-viscosity seed η̄ = (∂τ/∂η)ᵀ τ̄ / (1 − F), including the transpose of `harm_clamped`,
and adds (∂η/∂τ)ᵀ η̄ to the stress seeds, from local derivatives computed once per adjoint
solve (`implicit_viscosity_scratch`). It then reverses the stress kernel with the corrected
stress seeds.
"""
function enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt,
        args = (;),
    )
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt, args;
        τ_bar = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        τv_bar = (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
        τII_bar = adjoint.τ.II,
        ε_bar = (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        P_bar = adjoint.P,
        η_bar = adjoint.viscosity.η,
        θ_bar = adjoint.θ,
    )
    return nothing
end

# Select the same linear or implicit nonlinear constitutive pullback as the forward solve.
# Nonlinear solves allocate `constitutive_scratch`; linear solves pass `nothing`.
function enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt,
        args, viscosity_cutoff, constitutive_scratch; air_phase::Integer = 0,
    )
    fill!(adjoint.viscosity.η, 0)
    if isnothing(constitutive_scratch)
        enzyme_compute_stress_DRYEL!(
            stokes, adjoint, rheology, phase_ratios, ϕ, λ_relaxation, dt, args,
        )
    else
        enzyme_compute_variational_stress_DRYEL!(
            constitutive_scratch, stokes, adjoint, rheology, phase_ratios, ϕ,
            λ_relaxation, dt, args, viscosity_cutoff; air_phase,
        )
    end
    return nothing
end

function enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt, args;
        τ_bar = nothing, τv_bar = nothing, τII_bar = nothing, ε_bar = nothing,
        P_bar = nothing, η_bar = nothing, θ_bar = nothing,
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
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => θ_bar,
            ϕ => nothing,
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

"""
    implicit_viscosity_scratch(stokes, ϕ::JustRelax.RockRatio)

Storage of the implicit viscosity pullback of the variational DYREL adjoint, computed once per
adjoint solve and stress relaxation (`ensure_constitutive_tangent!`). There is no vertex
viscosity: a vertex stress reads the harmonic mean H of the four surrounding center
viscosities.

- `s`: ∂(τxx, τyy, τxy, τII, ΔPψ)/∂η at centers, `s_v`: ∂(τxx, τyy, τxy)/∂H at vertices;
- `v`: ∂η/∂(τxx, τyy, τxy), `vP`: ∂η/∂P; `feedback`: F = (∂η/∂τ)(∂τ/∂η) at centers.

`η` receives the implicit viscosity seed, which the material gradients reuse.
"""
function implicit_viscosity_scratch(stokes, ::JustRelax.RockRatio)
    center() = similar(stokes.viscosity.η)
    vertex() = similar(stokes.viscosity.ηv)
    return (;
        η_target = copy(stokes.viscosity.η),
        η = center(),
        feedback = center(),
        feedback_λ = Ref{Any}(nothing),
        s = ntuple(_ -> center(), 5),
        s_v = ntuple(_ -> vertex(), 3),
        v = ntuple(_ -> center(), 3),
        vP = center(),
        H = vertex(),
        H_seed = vertex(),
        direct = center(),
    )
end

# Harmonic mean of the center viscosities around a vertex, as read by the vertex stress.
@inline function vertex_viscosity_point!(H, η, periodic, I::Vararg{Integer, 2})
    Ic = clamped_indices(size(η), periodic, I...)
    @inbounds H[I...] = harm_clamped(η, Ic...)
    return nothing
end

@parallel_indices (I...) function vertex_viscosity!(H, η, periodic)
    vertex_viscosity_point!(H, η, periodic, I...)
    return nothing
end

# The vertex part of (∂τ/∂η)ᵀ τ̄, before it is spread to the centers.
@parallel_indices (I...) function vertex_viscosity_seed!(H̄, τ̄_v, s_v)
    @inbounds H̄[I...] = s_v[1][I...] * τ̄_v[1][I...] + s_v[2][I...] * τ̄_v[2][I...] +
        s_v[3][I...] * τ̄_v[3][I...]
    return nothing
end

function enzyme_compute_variational_stress_DRYEL!(
        scratch, stokes, adjoint, rheology, phase_ratios,
        ϕ::JustRelax.RockRatio, λ_relaxation, dt, args, cutoff;
        air_phase::Integer = 0,
    )
    enzyme_prepare_variational_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, cutoff; air_phase,
    )
    enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ, λ_relaxation, dt, args,
    )
    return nothing
end

function compute_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args, cutoff; air_phase::Integer = 0,
    )
    # ∂τ/∂η with the tangent η̇ = η: H is homogeneous of degree one, so then Ḣ = H, and the
    # center and vertex derivatives follow by division
    copyto!(scratch.η, stokes.viscosity.η)
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
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => scratch.s[5],
            ϕ => nothing,
            rheology => nothing,
            phase_ratios.center => nothing,
            phase_ratios.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            periodic_dims(stokes) => nothing,
        ),
    )
    @parallel (@idx size(scratch.H)) vertex_viscosity!(
        scratch.H, stokes.viscosity.η, periodic_dims(stokes),
    )
    foreach(A -> A ./= stokes.viscosity.η, scratch.s)
    foreach(A -> A ./= scratch.H, scratch.s_v)
    viscosity_tangent!(
        scratch.v, scratch.vP, scratch.η_target, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c), stokes, args, rheology, cutoff,
        local_viscosity_args; air_phase,
    )
    # the viscosity reads only center stresses, so the vertex stresses do not feed back
    compute_viscosity_feedback!(scratch.feedback, scratch.v, scratch.s)
    scratch.feedback_λ[] = λ_relaxation
    return nothing
end

# The outer and inner iterations may use different stress relaxations; recompute only then.
function ensure_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args, cutoff; air_phase::Integer = 0,
    )
    isequal(scratch.feedback_λ[], λ_relaxation) || compute_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt, args, cutoff;
        air_phase,
    )
    return nothing
end

function compute_constitutive_tangent_amplification!(
        scratch, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args, cutoff; air_phase::Integer = 0,
    )
    ensure_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, cutoff; air_phase,
    )
    feedback_max = maximum_mpi(scratch.feedback)
    isfinite(feedback_max) && feedback_max < 1 || error(
        "The nonlinear constitutive tangent is singular or unstable: maximum feedback = $feedback_max"
    )
    unit_amplification = one(eltype(scratch.η))
    @. scratch.η = max(unit_amplification, abs(inv(1 - scratch.feedback)))
    return nothing
end

# Form the corrected stress seeds and the implicit center-viscosity seed, which the material
# gradients reuse. The final reverse through the stress kernel is done by the caller.
function enzyme_prepare_variational_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phase_ratios,
        ϕ::JustRelax.RockRatio, λ_relaxation, dt, args, cutoff;
        air_phase::Integer = 0,
    )
    ensure_constitutive_tangent!(
        scratch, stokes, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, cutoff; air_phase,
    )
    # vertex stresses reach η through H: spread their part over the four centers
    @parallel (@idx size(scratch.H_seed)) vertex_viscosity_seed!(
        scratch.H_seed, (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy), scratch.s_v,
    )
    fill!(scratch.direct, 0)
    enzyme_reverse_pointwise!(
        vertex_viscosity_point!,
        size(scratch.H),
        (
            scratch.H => scratch.H_seed,
            stokes.viscosity.η => scratch.direct,
            periodic_dims(stokes) => nothing,
        );
        periodic = periodic_dims(stokes),
    )
    τ̄ = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c, adjoint.τ.II, adjoint.θ)
    @parallel (@idx size(scratch.η)) implicit_viscosity_seed!(
        scratch.η, scratch.direct, τ̄, scratch.s, scratch.v, adjoint.P, scratch.vP,
        scratch.feedback,
    )
    return nothing
end
