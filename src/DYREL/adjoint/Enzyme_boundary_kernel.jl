using Enzyme

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
