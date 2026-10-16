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

For nonlinear viscosity, `enzyme_compute_variational_stress_DRYEL!` performs three steps:

1. Reverse the stress kernel into scratch storage to collect the complete direct center-
   viscosity seed, including the transpose of `harm_clamped`.
2. Reverse viscosity with a unit seed and then stress with the resulting stress seed to obtain
   the scalar fixed-point feedback. Divide the direct seed by `1 - feedback`.
3. Reverse viscosity with the implicit seed and stress with the corrected stress seeds.
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
        ),
    )
    return nothing
end

# Step 1 of the nonlinear variational pullback: copy the incoming stress seeds and reverse only
# far enough to collect their complete derivative with respect to center viscosity. Because the
# forward vertex stress reads `harm_clamped(η)`, this reverse automatically gathers the four
# neighboring vertex contributions into each center. The real stress seeds remain unchanged for
# the final reverse sweep.
function enzyme_compute_variational_viscosity_seed!(
        scratch, stokes, adjoint, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args,
    )
    foreach(copyto!, scratch.τ, (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c))
    foreach(copyto!, scratch.τ_v, (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy))
    copyto!(scratch.τII, adjoint.τ.II)
    copyto!(scratch.θ, adjoint.θ)
    fill!(scratch.η, 0)
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt, args;
        τ_bar = scratch.τ, τv_bar = scratch.τ_v, τII_bar = scratch.τII,
        η_bar = scratch.η, θ_bar = scratch.θ,
    )
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

function enzyme_compute_viscosity_feedback_τII!(
        scratch, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args, cutoff; air_phase::Integer = 0,
    )
    zero_constitutive_stress_seed!(scratch)
    fill!(scratch.η, 1)
    enzyme_compute_viscosity_point!(
        scratch.η_target, scratch.η, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c), scratch.τ,
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args;
        air_phase,
    )
    fill!(scratch.η, 0)
    enzyme_reverse_stress_DRYEL!(
        stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt, args;
        τ_bar = scratch.τ, τv_bar = scratch.τ_v, τII_bar = scratch.τII,
        η_bar = scratch.η,
    )
    return nothing
end

function compute_constitutive_tangent_amplification!(
        scratch, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio,
        λ_relaxation, dt, args, cutoff; air_phase::Integer = 0,
    )
    enzyme_compute_viscosity_feedback_τII!(
        scratch, stokes, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, cutoff; air_phase,
    )
    feedback_max = maximum_mpi(scratch.η)
    isfinite(feedback_max) && feedback_max < 1 || error(
        "The nonlinear constitutive tangent is singular or unstable: maximum feedback = $feedback_max"
    )
    unit_amplification = one(eltype(scratch.η))
    @. scratch.η = max(unit_amplification, abs(inv(1 - scratch.η)))
    return nothing
end

# Form the corrected stress seeds and retain the implicit center-viscosity seed needed by the
# material-gradient pass. The final reverse through the stress kernel is done by the caller.
function enzyme_prepare_variational_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phase_ratios,
        ϕ::JustRelax.RockRatio, λ_relaxation, dt, args, cutoff;
        air_phase::Integer = 0,
    )
    # Direct path, including the transpose of the harmonic center-to-vertex interpolation.
    enzyme_compute_variational_viscosity_seed!(
        scratch, stokes, adjoint, rheology, phase_ratios, ϕ, λ_relaxation, dt, args,
    )
    copyto!(adjoint.viscosity.η, scratch.η)

    enzyme_compute_viscosity_feedback_τII!(
        scratch, stokes, rheology, phase_ratios, ϕ,
        λ_relaxation, dt, args, cutoff; air_phase,
    )
    # Apply the local scalar implicit-function correction. The feedback is a 1×1 quantity,
    # so it equals its transpose.
    @. adjoint.viscosity.η /= 1 - scratch.η

    # Preserve the implicit viscosity seed for material gradients before its reverse pass is
    # consumed, then add the nonlinear viscosity path to the original center-stress seed.
    copyto!(scratch.η, adjoint.viscosity.η)
    enzyme_compute_viscosity_point!(
        scratch.η_target, adjoint.viscosity.η, phase_ratios.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        args, rheology, cutoff, compute_viscosity_τII, local_viscosity_args;
        air_phase,
    )
    fill!(adjoint.viscosity.η, 0)
    return nothing
end
