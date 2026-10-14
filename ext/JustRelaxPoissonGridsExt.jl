module JustRelaxPoissonGridsExt

using JustRelax
using PoissonGrids

JustRelax._solve_refined_axis(lo, hi, monitor, ni) = PoissonGrids.solve_grid(lo, hi, monitor, ni)

end
