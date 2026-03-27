# Grid generation

JustRelax uses staggered Cartesian grids. The main entry point is [`Geometry`](@ref), which stores:

- `xci`: cell-centered coordinates
- `xvi`: vertex coordinates
- `xi_vel`: staggered velocity coordinates
- `di`: grid spacing at cell centers, vertices, and velocity locations

For most workflows you either build a uniform grid from the number of cells and domain size, or a nonuniform grid from explicit vertex coordinates.

## Uniform grids

Use `Geometry(ni, li; origin = ...)` to create a uniform grid:

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

```julia
using JustRelax

xv = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
yv = [-1.0, -0.7, -0.45, -0.2, 0.0]

grid = Geometry(xv, yv)

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

If you want the coordinate arrays stored in a specific array type, pass an array constructor as the first argument:

```julia
grid = Geometry(Array, xv, yv)
```

## MPI-distributed grids

For distributed runs, initialize `ImplicitGlobalGrid` first and then construct the grid exactly as in serial:

```julia
using JustRelax

nx, ny = 128, 64
igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)

grid = Geometry((nx, ny), (1.0e6, 3.0e5); origin = (0.0, -3.0e5))
```

Here `grid.xci`, `grid.xvi`, and `grid.xi_vel` correspond to the local rank, while the spacing is computed from the global grid dimensions returned by `ImplicitGlobalGrid`.

## API reference

```@docs; canonical=false
IGG
Geometry
lazy_grid
velocity_grids
```
