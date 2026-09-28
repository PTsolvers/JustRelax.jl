using Enzyme

"""
    enzyme_compute_∇V_strain_rate_RP!(
        stokes, adjoint, dyrel, rheology, phase_ratios, _di, ni, dt, args;
        do_strain_rate=true,
    )

`adjoint.ε` and `adjoint.R.RP` are the reverse seeds for the strain-rate and
pressure-residual outputs. The resulting velocity and pressure sensitivities
accumulate in `adjoint.V` and `adjoint.P`; all remaining inputs are constant.
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

Differentiate the two-dimensional Powell–Hestenes momentum-residual kernel.
`adjoint.R.Rx` and `adjoint.R.Ry` seed the residual outputs. Velocity,
pressure, stress, and buoyancy sensitivities accumulate in their corresponding
adjoint fields.
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

Reverse the momentum residual and propagate its buoyancy sensitivity through density
to the coupled Stokes pressure. The adjoint solver requires `args.P === stokes.P`.
"""
function enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, _di, ni, rheology, phases, args; free_surface_dt = 0
    )
    zero_buoyancy_adjoint!(adjoint)
    enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, _di, ni; free_surface_dt)
    buoyancy_pressure_adjoint!(adjoint, phases, rheology, args, ni)
    return nothing
end

# The buoyancy adjoints collect the residual's ρg seeds, so they start from zero on every pass.
zero_buoyancy_adjoint!(adjoint) = foreach(A -> fill!(A, 0.0), (adjoint.dρgx, adjoint.ρ))

"""
    buoyancy_pressure_adjoint!(adjoint, phases, rheology, args, ni; air_phase=0)

Propagate the buoyancy adjoints `adjoint.dρgx`/`adjoint.ρ` through the pressure dependence
of density into `adjoint.P`. `air_phase` must match the forward buoyancy update, which drops
that phase from the density average; `0` keeps every phase.
"""
function buoyancy_pressure_adjoint!(adjoint, phases, rheology, args, ni; air_phase::Integer = 0)
    @parallel (@idx ni) buoyancy_pressure_adjoint_kernel!(
        adjoint.P, (adjoint.dρgx, adjoint.ρ), phases.center, rheology, args, air_phase
    )
    return nothing
end

@inline function density_at_pressure(P, rheology, ratio, args)
    return fn_ratio(compute_density, rheology, ratio, merge(args, (; P)))
end

# The phase ratio is corrected for `air_phase` exactly as in `compute_ρg_kernel!`; an all-air
# cell ends up with an all-zero ratio and contributes nothing.
@parallel_indices (I...) function buoyancy_pressure_adjoint_kernel!(
        dP, dρg, phases, rheology, args, air_phase::Integer
    )
    local_args = getindex_NamedTuple(args, I...)
    ratio = correct_phase_ratio(air_phase, @cell phases[I...])
    dρdP = Enzyme.autodiff_deferred(
        Enzyme.Reverse,
        Enzyme.Const(density_at_pressure),
        Enzyme.Active,
        Enzyme.Active(local_args.P),
        Enzyme.Const(rheology),
        Enzyme.Const(ratio),
        Enzyme.Const(local_args),
    )[1][1]
    gravity = compute_gravity(first(rheology))
    gx, gy = gravity isa Number ? (zero(gravity), gravity) : (gravity[1], gravity[3])
    # Momentum -> buoyancy -> density -> pressure. Add to the direct pressure
    # gradient and any objective seed already present in dP.
    dP[I...] += dρdP * (gx * dρg[1][I...] + gy * dρg[2][I...])
    return nothing
end

"""
    enzyme_compute_stress_viscosity_DRYEL!(
        stokes, adjoint, θc, γ_eff, rheology, phase_ratios, λ_relaxation, dt,
        viscosity_relaxation, args, viscosity_cutoff, linear_viscosity,
    )

Differentiate the same fused stress and viscosity-update kernel used by the
two-dimensional forward solver. Stress and viscosity adjoints are propagated
to strain rate, pressure, and the previous viscosity iterate.
"""
function enzyme_compute_stress_viscosity_DRYEL!(
        stokes,
        adjoint,
        θc,
        γ_eff,
        rheology,
        phase_ratios,
        λ_relaxation,
        dt,
        viscosity_relaxation,
        args,
        viscosity_cutoff,
        linear_viscosity,
    )
    periodic = periodic_dims(stokes)
    enzyme_reverse_pointwise!(
        compute_stress_viscosity_DRYEL_point!,
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
            stokes.λ => nothing,
            stokes.λv => nothing,
            stokes.viscosity.η => adjoint.viscosity.η,
            stokes.viscosity.ηv => adjoint.viscosity.ηv,
            stokes.viscosity.η_vep => nothing,
            stokes.ΔPψ => adjoint.θ,
            θc => nothing,
            stokes.R.RP => nothing,
            γ_eff => nothing,
            rheology => nothing,
            phase_ratios.center => nothing,
            phase_ratios.vertex => nothing,
            λ_relaxation => nothing,
            dt => nothing,
            viscosity_relaxation => nothing,
            args => nothing,
            viscosity_cutoff => nothing,
            linear_viscosity => nothing,
            periodic => nothing,
        ),
    )
    return nothing
end

"""
    enzyme_flow_bcs!(stokes, adjoint, bcs)

Reverse the two-dimensional velocity boundary kernels. Velocity sensitivities
are accumulated in `adjoint.V`.
"""
function enzyme_flow_bcs!(stokes, adjoint, bcs)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = adjoint.V.Vx, adjoint.V.Vy
    n = bc_index((Vx, Vy))

    # Reverse the order used by `_flow_bcs!`.
    if do_bc(bcs.periodic)
        @parallel (@idx n) configcall = periodic_boundary!(Vx, Vy, bcs.periodic) ParallelStencil.AD.autodiff_deferred!(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            periodic_boundary!,
            Enzyme.Duplicated(Vx, dVx),
            Enzyme.Duplicated(Vy, dVy),
            Enzyme.Const(bcs.periodic),
        )
    end
    if do_bc(bcs.free_slip)
        @parallel (@idx n) configcall = free_slip!(Vx, Vy, bcs.free_slip) ParallelStencil.AD.autodiff_deferred!(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            free_slip!,
            Enzyme.Duplicated(Vx, dVx),
            Enzyme.Duplicated(Vy, dVy),
            Enzyme.Const(bcs.free_slip),
        )
    end
    if do_bc(bcs.no_slip)
        enzyme_no_slip!(Vx, dVx, Vy, dVy, bcs.no_slip)
    end
    return nothing
end

"""
    enzyme_no_slip!(Vx, dVx, Vy, dVy, bc)

Reverse the active two-dimensional no-slip boundary kernels. The four kernels
are called in the reverse order of `no_slip!`.
"""
function enzyme_no_slip!(Vx, dVx, Vy, dVy, bc)
    n1 = max(size(Vx, 2), size(Vy, 2))
    n2 = max(size(Vx, 1), size(Vy, 1))
    bc.top && _enzyme_no_slip_top!(Vx, dVx, Vy, dVy, n2)
    bc.bot && _enzyme_no_slip_bot!(Vx, dVx, Vy, dVy, n2)
    bc.right && _enzyme_no_slip_right!(Vx, dVx, Vy, dVy, n1)
    bc.left && _enzyme_no_slip_left!(Vx, dVx, Vy, dVy, n1)
    return nothing
end

function _enzyme_no_slip_left!(Vx, dVx, Vy, dVy, n)
    @parallel (@idx n) configcall = _no_slip_left!(Vx, Vy) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _no_slip_left!,
        Enzyme.Duplicated(Vx, dVx),
        Enzyme.Duplicated(Vy, dVy),
    )
    return nothing
end

function _enzyme_no_slip_right!(Vx, dVx, Vy, dVy, n)
    @parallel (@idx n) configcall = _no_slip_right!(Vx, Vy) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _no_slip_right!,
        Enzyme.Duplicated(Vx, dVx),
        Enzyme.Duplicated(Vy, dVy),
    )
    return nothing
end

function _enzyme_no_slip_bot!(Vx, dVx, Vy, dVy, n)
    @parallel (@idx n) configcall = _no_slip_bot!(Vx, Vy) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _no_slip_bot!,
        Enzyme.Duplicated(Vx, dVx),
        Enzyme.Duplicated(Vy, dVy),
    )
    return nothing
end

function _enzyme_no_slip_top!(Vx, dVx, Vy, dVy, n)
    @parallel (@idx n) configcall = _no_slip_top!(Vx, Vy) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        _no_slip_top!,
        Enzyme.Duplicated(Vx, dVx),
        Enzyme.Duplicated(Vy, dVy),
    )
    return nothing
end
