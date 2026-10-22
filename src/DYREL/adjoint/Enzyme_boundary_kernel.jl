using Enzyme

"""
    enzyme_flow_bcs!(stokes, adjoint, bcs)

Reverse the two-dimensional velocity boundary kernels. The velocity output seeds and the
resulting reverse-mode derivatives are stored in `adjoint.V`.
"""
function enzyme_flow_bcs!(stokes, adjoint, bcs)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = adjoint.V.Vx, adjoint.V.Vy
    n = bc_index((Vx, Vy))

    # Reverse the order used by `_flow_bcs!`.
    if do_bc(bcs.periodic)
        enzyme_reverse_pointwise!(
            periodic_boundary_point!, n, (Vx => dVx, Vy => dVy, bcs.periodic => nothing)
        )
    end
    if do_bc(bcs.free_slip)
        enzyme_reverse_pointwise!(
            free_slip_point!, n, (Vx => dVx, Vy => dVy, bcs.free_slip => nothing)
        )
    end
    if do_bc(bcs.no_slip)
        enzyme_no_slip!(Vx, dVx, Vy, dVy, bcs.no_slip)
    end
    return nothing
end

"""
    enzyme_no_slip!(Vx, dVx, Vy, dVy, bc)

Reverse the active two-dimensional no-slip boundary kernels. `dVx` and `dVy` provide the
velocity output seeds and receive the resulting reverse-mode derivatives. The four kernels are
called in the reverse order of `no_slip!`.
"""
function enzyme_no_slip!(Vx, dVx, Vy, dVy, bc)
    n1 = max(size(Vx, 2), size(Vy, 2))
    n2 = max(size(Vx, 1), size(Vy, 1))
    V = (Vx => dVx, Vy => dVy)
    bc.top && enzyme_reverse_pointwise!(_no_slip_top_point!, n2, V)
    bc.bot && enzyme_reverse_pointwise!(_no_slip_bot_point!, n2, V)
    bc.right && enzyme_reverse_pointwise!(_no_slip_right_point!, n1, V)
    bc.left && enzyme_reverse_pointwise!(_no_slip_left_point!, n1, V)
    return nothing
end
