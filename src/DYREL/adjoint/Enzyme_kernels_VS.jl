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
strain-rate and pressure-residual outputs; velocity and pressure sensitivities accumulate in
`adjoint.V` and `adjoint.P`. The rock-fraction scaling of the continuity residual is
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
`adjoint.R.Rx` and `adjoint.R.Ry` seed the residual outputs. Velocity, pressure, stress and
buoyancy sensitivities accumulate in their corresponding adjoint fields.
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

Reverse the masked momentum residual and propagate its buoyancy sensitivity through density
to the coupled Stokes pressure. `air_phase` must match the forward `update_ρg!` call, which
drops that phase from the density average. The adjoint solver requires `args.P === stokes.P`.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, ϕ::JustRelax.RockRatio, _di, ni, rheology, phases, args;
        free_surface_dt = 0, air_phase::Integer = 0,
    )
    zero_buoyancy_adjoint!(adjoint)
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, ϕ, _di, ni; free_surface_dt)
    buoyancy_pressure_adjoint!(adjoint, phases, rheology, args, ni; air_phase)
    return nothing
end

"""
    enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ::RockRatio, λ_relaxation, dt, args,
    )

Differentiate the masked DYREL constitutive kernel. Stress adjoints in `adjoint.τ` are
propagated to strain rate, pressure and the plastic pressure correction (`adjoint.ε`,
`adjoint.P`, `adjoint.θ`).

The variational kernel has no vertex viscosity of its own: it evaluates `harm_clamped(η)` of
the surrounding centers. With `η` active, the vertex contributions are therefore folded into
the center field `adjoint.viscosity.η` through the harmonic mean, and `adjoint.viscosity.ηv`
is left untouched. `adjoint.viscosity.η` accumulates across calls, so the caller zeroes it
before the pass whose viscosity sensitivity it wants to keep.

The separate τII-viscosity refresh (`update_viscosity_τII!`) is not differentiated here, so
this covers `linear_viscosity = true`.
"""
function enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ::JustRelax.RockRatio, λ_relaxation, dt,
        args = (;),
    )
    periodic = periodic_dims(stokes)
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
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => adjoint.θ,
            ϕ => nothing,
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
