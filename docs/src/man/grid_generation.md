# Grid generation

JustRelax uses staggered Cartesian grids. The main entry point is [`Geometry`](@ref), which stores:

- `xci`: cell-centered coordinates
- `xvi`: vertex coordinates
- `xi_vel`: staggered velocity coordinates
- `di`: grid spacing at cell centers, vertices, and velocity locations

For most workflows you either build a uniform grid from the number of cells and domain size, or a nonuniform grid from explicit vertex coordinates.

## Uniform grids

Use `Geometry(ni, li; origin = ...)` to create a uniform grid:

<!-- Not an @example block: the Getting started page runs init_global_grid, so a
     Geometry(ni, li) evaluated during the build reports the spacing of that global
     grid rather than of ni. -->

```julia
using JustRelax

ni = (128, 64)                  # number of cells
li = (1.0e6, 3.0e5)             # physical domain size
origin = (0.0, -3.0e5)

grid = Geometry(ni, li; origin = origin)

xci = grid.xci                  # cell-centered coordinates
xvi = grid.xvi                  # vertex coordinates
grid_vx, grid_vy = grid.xi_vel  # staggered velocity coordinates

dx, dy = grid.di.center
```

In serial, the grid covers the full domain directly. If `ImplicitGlobalGrid` is already initialized, the same constructor returns the local MPI subdomain, while still using the global domain lengths `li` to compute the spacing.

## Nonuniform grids

Use explicit vertex coordinates when you want local refinement or nonuniform spacing:

```@example grid
using JustRelax

xv = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
yv = [-1.0, -0.7, -0.45, -0.2, 0.0]

grid = Geometry((xv, yv))

xci = grid.xci
xvi = grid.xvi
grid_vx, grid_vy = grid.xi_vel

dx = grid.di.vertex[1]
dy = grid.di.vertex[2]
```

This constructor derives:

- cell-centered coordinates from the vertex coordinates
- nonuniform spacings with `diff.(xvi)`
- staggered velocity grids with the required ghost points

The 3D Stokes solvers (`solve!` and `solve_VariationalStokes!`) support only uniform grids and throw an `ArgumentError` when given a nonuniform 3D `Geometry`.

The vertex coordinates are passed as a tuple. If you want the coordinate arrays stored in a specific array type, pass an array constructor as the first argument, in which case the coordinates are given as separate arguments:

```@example grid
grid = Geometry(Array, xv, yv)
grid.xvi[1]
```

## PoissonGrids refinement

Loading the optional PoissonGrids package activates monitor-based geometry
construction. Supply one monitor per axis; `nothing` keeps an axis uniform:

```julia
using JustRelax, PoissonGrids

M = window_monitor(5.0, 0.25, 32.5, -27.5)
grid = Geometry(Array, (112, 64), (450.0, 210.0), (nothing, M);
    origin = (-225.0, -200.0))
```

The monitor uses the same coordinate units as `li` and `origin` (km here).
`Geometry` calls `PoissonGrids.solve_grid` internally and constructs the center,
vertex, velocity, and spacing arrays. Omitting `Array` selects CPU arrays;
`PTArray(backend)` selects another backend's array type. Monitor validation and
convergence errors propagate from PoissonGrids.

With `igg` supplied, `ni` counts **local** cells, matching `init_global_grid`.
The constructor generates global coordinates and partitions them internally:

```julia
ni = (58, 34)
igg = IGG(init_global_grid(ni..., 1; select_device = false)...)
grid = Geometry(Array, ni, (450.0, 210.0), (nothing, M);
    origin = (-225.0, -200.0), igg)
```

Without `igg`, this constructor always returns the full serial geometry,
even when an IGG topology exists. This allows a global reference grid to be
constructed alongside the distributed one. Nonuniform 3D geometry construction
is supported; the solver restrictions described above still apply.

## MPI-distributed grids

For distributed runs, initialize `ImplicitGlobalGrid` first and then construct the grid exactly as in serial:

```julia
using JustRelax

nx, ny = 128, 64
igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)

grid = Geometry((nx, ny), (1.0e6, 3.0e5); origin = (0.0, -3.0e5))
```

Here `grid.xci`, `grid.xvi`, and `grid.xi_vel` correspond to the local rank, while the spacing is computed from the global grid dimensions returned by `ImplicitGlobalGrid`.

For nonuniform coordinates generated elsewhere, pass the **global** vertex
vectors explicitly:

```julia
grid = Geometry(Array, (xv_global, yv_global), igg)
```

Both distributed nonuniform constructors require nonperiodic boundaries and the
default two-cell overlap. They preserve global domain lengths for solver scaling
and use neighboring global coordinates for internal velocity ghost nodes. The
plain `Geometry(Array, xv, yv)` constructor continues to interpret its vectors as
local coordinates without partitioning. Parallel output can use the explicit
coordinates as described in [Visualization](./visualization.md).

## Particle initialization

Particles are initialized from the staggered velocity grids stored in `Geometry`:

```julia
using JustPIC

nxcell = 24
max_xcell = 36
min_xcell = 12

particles = init_particles(backend, nxcell, max_xcell, min_xcell, grid.xi_vel...)
```

`velocity_grids(xci, xvi, di)` builds the same coordinates outside of a `Geometry`, for the cases that need them separately.

## API reference

```@docs; canonical=false
IGG
Geometry
lazy_grid
velocity_grids
```
