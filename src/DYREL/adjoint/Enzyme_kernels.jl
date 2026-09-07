using Enzyme

"""
    enzyme_compute_∇V_strain_rate_RP!(
        stokes, adjoint, dyrel, rheology, phase_ratios, _di, ni, dt, args;
        do_strain_rate=true,
    )

Example reverse-mode differentiation of the two-dimensional
`compute_∇V_strain_rate_RP!` ParallelStencil kernel.

`adjoint.ε` and `adjoint.PA` are the reverse seeds for the strain-rate and
pressure-residual outputs. The resulting velocity and pressure sensitivities
accumulate in `adjoint.VA` and `adjoint.P`; all remaining inputs are constant.
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

    @parallel (@idx ni .+ 1) configcall = compute_∇V_strain_rate_RP!(
        stokes.ε.xx,
        stokes.ε.yy,
        stokes.ε.xy,
        stokes.V.Vx,
        stokes.V.Vy,
        stokes.R.RP,
        stokes.P,
        stokes.P0,
        stokes.Q,
        dyrel.ηb,
        _di.vertex,
        _di.velocity...,
        rheology,
        phase_ratios.center,
        ΔT,
        melt_fraction,
        dt,
        do_strain_rate,
    ) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        compute_∇V_strain_rate_RP!,
        Enzyme.DuplicatedNoNeed(stokes.ε.xx, adjoint.ε.xx),
        Enzyme.DuplicatedNoNeed(stokes.ε.yy, adjoint.ε.yy),
        Enzyme.DuplicatedNoNeed(stokes.ε.xy, adjoint.ε.xy),
        Enzyme.DuplicatedNoNeed(stokes.V.Vx, adjoint.VA.Vx),
        Enzyme.DuplicatedNoNeed(stokes.V.Vy, adjoint.VA.Vy),
        Enzyme.DuplicatedNoNeed(stokes.R.RP, adjoint.PA),
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
`adjoint.R.Rx` and `adjoint.R.Ry` seed the residual outputs; pressure and
stress sensitivities accumulate in `adjoint.P`, `adjoint.θ`, and
`adjoint.dτ`. Buoyancy is constant in this pass.
"""
function enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, _di, ni)
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
        Enzyme.DuplicatedNoNeed(stokes.τ.xx, adjoint.dτ.xx),
        Enzyme.DuplicatedNoNeed(stokes.τ.yy, adjoint.dτ.yy),
        Enzyme.DuplicatedNoNeed(stokes.τ.xy, adjoint.dτ.xy),
        Enzyme.Const(ρg[1]),
        Enzyme.Const(ρg[2]),
        Enzyme.Const(_di.center),
        Enzyme.Const(_di.vertex),
    )
    return nothing
end

"""
    enzyme_compute_PH_residual_V_sensitivity!(stokes, adjoint, ρg, dρg, _di, ni)

Differentiate the momentum residual only with respect to the two buoyancy-force
arrays. The residual seeds are read from `adjoint.R`; sensitivities accumulate
in the matching arrays of `dρg`.
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
        Enzyme.Const(stokes.P),
        Enzyme.Const(stokes.ΔPψ),
        Enzyme.Const(stokes.τ.xx),
        Enzyme.Const(stokes.τ.yy),
        Enzyme.Const(stokes.τ.xy),
        Enzyme.DuplicatedNoNeed(ρg[1], dρg[1]),
        Enzyme.DuplicatedNoNeed(ρg[2], dρg[2]),
        Enzyme.Const(_di.center),
        Enzyme.Const(_di.vertex),
    )
    return nothing
end

"""
    enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt,
    )

Differentiate the two-dimensional DYREL constitutive kernel. Stress adjoints in
`adjoint.dτ` are propagated to strain rate and pressure in `adjoint.ε` and
`adjoint.P`. This is the production equivalent of `diff_calc_stress!` in the
toy example.
"""
function enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt
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
    ) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        compute_stress_DRYEL!,
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
            (adjoint.dτ.xx, adjoint.dτ.yy, adjoint.dτ.xy_c),
        ),
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
            (adjoint.dτ.xx_v, adjoint.dτ.yy_v, adjoint.dτ.xy),
        ),
        Enzyme.Const((stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c)),
        Enzyme.Const((stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy)),
        Enzyme.DuplicatedNoNeed(stokes.τ.II, adjoint.dτ.II),
        Enzyme.DuplicatedNoNeed(
            (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy),
            (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
        ),
        Enzyme.Const((stokes.ε_pl.xx, stokes.ε_pl.yy, stokes.ε_pl.xy)),
        Enzyme.Const(stokes.EII_pl),
        Enzyme.Const(stokes.ε_vol_pl),
        Enzyme.DuplicatedNoNeed(stokes.P, adjoint.P),
        Enzyme.Const(stokes.λ),
        Enzyme.Const(stokes.λv),
        Enzyme.Const(stokes.viscosity.η),
        Enzyme.Const(stokes.viscosity.ηv),
        Enzyme.Const(stokes.viscosity.η_vep),
        Enzyme.DuplicatedNoNeed(stokes.ΔPψ, adjoint.θ),
        Enzyme.Const(rheology),
        Enzyme.Const(phase_ratios.center),
        Enzyme.Const(phase_ratios.vertex),
        Enzyme.Const(λ_relaxation),
        Enzyme.Const(dt),
    )
    return nothing
end

"""
    enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt,
    )

Differentiate the constitutive kernel only with respect to center and vertex
viscosity. Stress seeds come from `adjoint.dτ`; viscosity sensitivities
accumulate in `adjoint.viscosity.η` and `adjoint.viscosity.ηv`.
"""
function enzyme_compute_stress_DRYEL_sensitivity!(
        stokes, adjoint, rheology, phase_ratios, λ_relaxation, dt
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
    ) ParallelStencil.AD.autodiff_deferred!(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        compute_stress_DRYEL!,
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
            (adjoint.dτ.xx, adjoint.dτ.yy, adjoint.dτ.xy_c),
        ),
        Enzyme.DuplicatedNoNeed(
            (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy),
            (adjoint.dτ.xx_v, adjoint.dτ.yy_v, adjoint.dτ.xy),
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
    )
    return nothing
end

"""
    enzyme_flow_bcs!(stokes, adjoint, bcs)

Reverse the two-dimensional velocity boundary kernels. Velocity sensitivities
are accumulated in `adjoint.VA`.
"""
function enzyme_flow_bcs!(stokes, adjoint, bcs)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = adjoint.VA.Vx, adjoint.VA.Vy
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
