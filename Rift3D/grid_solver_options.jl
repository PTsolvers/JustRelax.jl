"""
    grid_solver_options(nc)

Resolution-dependent options for PoissonGrids' 1D iterative solver with the Rift
window monitor. `nc` counts cells along the refined axis, independently of the
horizontal resolution or MPI decomposition.

Keep the original tolerance through 512 cells. Above that, a cubic scaling gives
the residual a margin above the observed Float64 noise at high resolution. The
iteration budget grows linearly with resolution. These choices are benchmarked
for the [-200, 10] km domain and `window_monitor(5, 0.25, 32.5, -27.5)`;
different coordinate units or monitors may require different settings.
"""
function grid_solver_options(nc::Integer)
    nc > 0 || throw(ArgumentError("nc must be positive"))
    tol = 1.0e-7 * max(1.0, nc / 512)^3
    maxiter = max(20_000, 50 * nc)
    return (; tol, maxiter)
end
