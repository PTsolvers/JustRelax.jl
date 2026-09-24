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

    enzyme_reverse_rowwise!(
        compute_∇V_strain_rate_RP_point!,
        ni .+ 1,
        Enzyme.DuplicatedNoNeed(stokes.ε.xx, adjoint.ε.xx),
        Enzyme.DuplicatedNoNeed(stokes.ε.yy, adjoint.ε.yy),
        Enzyme.DuplicatedNoNeed(stokes.ε.xy, adjoint.ε.xy),
        Enzyme.DuplicatedNoNeed(stokes.V.Vx, adjoint.V.Vx),
        Enzyme.DuplicatedNoNeed(stokes.V.Vy, adjoint.V.Vy),
        Enzyme.DuplicatedNoNeed(stokes.R.RP, adjoint.R.RP),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.Const(stokes.P0),
        Enzyme.Const(stokes.Q),
        Enzyme.Const(dyrel.ηb),
        Enzyme.Const(_di.vertex),
        Enzyme.Const(_di.velocity[1]),
        Enzyme.Const(_di.velocity[2]),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios.center),
        Enzyme.Const(ΔT),
        Enzyme.Const(melt_fraction),
        Enzyme.Const(dt),
        Enzyme.Const(do_strain_rate),
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
    enzyme_reverse_rowwise!(
        compute_PH_residual_V_point!,
        ni,
        Enzyme.DuplicatedNoNeed(stokes.R.Rx, adjoint.R.Rx),
        Enzyme.DuplicatedNoNeed(stokes.R.Ry, adjoint.R.Ry),
        Enzyme.DuplicatedNoNeed(stokes.V.Vx, adjoint.V.Vx),
        Enzyme.DuplicatedNoNeed(stokes.V.Vy, adjoint.V.Vy),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.DuplicatedNoNeed(stokes.τ.xx, adjoint.τ.xx),
        Enzyme.DuplicatedNoNeed(stokes.τ.yy, adjoint.τ.yy),
        Enzyme.DuplicatedNoNeed(stokes.τ.xy, adjoint.τ.xy),
        Enzyme.DuplicatedNoNeed(ρg[1], adjoint.dρgx),
        Enzyme.DuplicatedNoNeed(ρg[2], adjoint.ρ),
        Enzyme.Const(_di.center),
        Enzyme.Const(_di.vertex),
        Enzyme.Const(free_surface_dt),
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
    enzyme_reverse_rowwise!(
        compute_stress_viscosity_DRYEL_point!,
        size(phase_ratios.vertex),
        Enzyme.DuplicatedNoNeed((stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),(adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c)),
        Enzyme.DuplicatedNoNeed((stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),(adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy)),
        Enzyme.Const((stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c)),
        Enzyme.Const((stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy)),
        Enzyme.DuplicatedNoNeed(stokes.τ.II, adjoint.τ.II),
        Enzyme.DuplicatedNoNeed((stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),(adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy)),
        Enzyme.Const((stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)),
        Enzyme.Const(stokes.EII_pl),
        Enzyme.Const(stokes.ε_vol_pl),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.Const(stokes.λ),
        Enzyme.Const(stokes.λv),
        Enzyme.DuplicatedNoNeed(stokes.viscosity.η, adjoint.viscosity.η),
        Enzyme.DuplicatedNoNeed(stokes.viscosity.ηv, adjoint.viscosity.ηv),
        Enzyme.Const(stokes.viscosity.η_vep),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.Const(θc),
        Enzyme.Const(stokes.R.RP),
        Enzyme.Const(γ_eff),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios.center),
        Enzyme.Const(phase_ratios.vertex),
        Enzyme.Const(λ_relaxation),
        Enzyme.Const(dt),
        Enzyme.Const(viscosity_relaxation),
        Enzyme.Const(args),
        Enzyme.Const(viscosity_cutoff),
        Enzyme.Const(linear_viscosity),
        Enzyme.Const(periodic),
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
