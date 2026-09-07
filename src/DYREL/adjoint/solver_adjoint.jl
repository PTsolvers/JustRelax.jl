"""
    solve_DYREL_adjoint!(backend, stokes, args...; kwargs...)

Adjoint entry point called by `solve_DYREL!(...; adjoint=true)` after the
forward iterations converge and before history-dependent state is updated.
Implement backend-specific methods here.
"""
function solve_DYREL_adjoint! end
