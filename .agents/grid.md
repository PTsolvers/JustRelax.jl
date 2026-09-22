# Grid and field layout

Companion to [`AGENTS.md`](../AGENTS.md). Covers the grid objects, where each field lives
on the staggered mesh, and how spacing is addressed in kernels. See [`api.md`](api.md) for
constructors, [`solvers.md`](solvers.md) for what consumes them, and [`mpi.md`](mpi.md) for
the distributed case.

Source: [`src/grid/`](../src/grid/) — `Grid.jl` (entry, `IGG`, coordinate builders),
`Cartesian.jl` (`Geometry`), `Annulus.jl` (`GeometryAnnulus`), `Utils.jl` (`x_g`/`y_g`/`z_g`
and the spacing macros).

## Building a grid

```julia
ni     = nx, ny                    # number of CELLS
li     = lx, ly                    # physical extent
origin = 0.0, -ly
grid   = Geometry(ni, li; origin)
(; xci, xvi) = grid
```

`Geometry(ni, li; origin)` builds a uniform staggered grid. It checks
`ImplicitGlobalGrid.grid_is_initialized()`: with a global grid initialized it goes through
`geometry_MPI`, deriving `di` from the *global* cell counts and returning coordinates for
the **local** subdomain; otherwise `geometry_nonMPI` covers the whole domain.

`Geometry(TA, xv1, xv2[, xv3])` builds a nonuniform grid from explicit vertex-coordinate
vectors, with `TA` the array constructor used to materialize the coordinates (`Array`, or a
device array type). Centers, per-cell spacings, and velocity grids are derived from the
vertices. `Geometry(xvi::NTuple)` is the `Array` shorthand.

`GeometryAnnulus` is the same structure with `(θ, r)` coordinate order: `xci[1]`/`xvi[1]`
are angles in radians, `xci[2]`/`xvi[2]` radii. Both share the supertype
`AbstractGrid{nDim, V, D, T}`.

## `Geometry` fields

| Field | Meaning |
|-------|---------|
| `ni` | number of local cells per direction |
| `li` | domain lengths |
| `origin` | lower corner |
| `max_li` | `max(li...)`, the length scale used in PT coefficients and residual scaling |
| `di`, `_di` | spacing and its reciprocal (see below) |
| `xci` | cell-center coordinates, `length ni[d]` |
| `xvi` | vertex coordinates, `length ni[d] + 1` |
| `xi_vel` | one coordinate tuple per velocity component |

`di` and `_di` are **NamedTuples with three entries**, not plain tuples:

```julia
grid.di.center      # spacing between cell centers
grid.di.vertex      # spacing between vertices
grid.di.velocity    # one tuple per velocity component
grid._di.vertex     # inv.() of the above, precomputed
```

Kernels take `_di` and select the entry matching the field they differentiate — e.g.
`compute_∇V!` gets `_di.vertex`, the residual kernels get both `_di.center` and
`_di.vertex`. For a uniform grid every entry holds scalars; for a nonuniform grid they hold
vectors. Legacy call forms that accept a bare `di` tuple accept `grid.di.center`.

`xi_vel` gives the coordinates for particle interpolation: for each velocity component, the
coordinate along that component sits on vertices while the transverse directions are
extended by one ghost point on each side.

## Where fields live

With `ni = (nx, ny)` cells in 2D:

| Field | Size | Location |
|-------|------|----------|
| `stokes.P`, `P0`, `∇V`, `Q`, `λ`, `EII_pl`, `ΔPψ` | `(nx, ny)` | centers |
| `stokes.V.Vx` | `(nx+1, ny+2)` | x-faces, one ghost row |
| `stokes.V.Vy` | `(nx+2, ny+1)` | y-faces, one ghost column |
| `stokes.U.Ux`, `Uy` | as `Vx`, `Vy` | displacement, same staggering |
| `stokes.τ.xx`, `yy`, `xy_c`, `II` | `(nx, ny)` | centers |
| `stokes.τ.xx_v`, `yy_v`, `xy` | `(nx+1, ny+1)` | vertices |
| `stokes.viscosity.η`, `η_vep`, `ητ` | `(nx, ny)` | centers |
| `stokes.viscosity.ηv`, `stokes.λv` | `(nx+1, ny+1)` | vertices |
| `stokes.ω.xy` | `(nx+1, ny+1)` | vertices |
| `thermal.T`, `Told`, `ΔT` | `(nx+2, ny+2)` | centers **plus one ghost node per boundary** |
| `thermal.H`, `shear_heating`, `adiabatic`, `dT_dt`, `ResT` | `(nx, ny)` | centers |
| `thermal.qTx`, `qTx2` | `(nx+1, ny)` | x-faces |
| `thermal.qTy`, `qTy2` | `(nx, ny+1)` | y-faces |

In 3D: `Vx` is `(nx+1, ny+2, nz+2)` and cyclic permutations; `τ.xy` is `(nx+1, ny+1, nz)`,
`τ.yz` is `(nx, ny+1, nz+1)`, `τ.xz` is `(nx+1, ny, nz+1)`; the `_v` normal components and
`ω` components follow the same pattern; `T` is `ni .+ 2`.

`thermal.T` is the trap worth remembering: it is cell-centered with a ghost ring, so
`thermal.T[2:end-1, 2:end-1]` is the interior field aligned with `xci`, and a kernel over
it is launched with `@parallel (@idx size(T) .- 2)`.

## Spacing macros

`src/grid/Utils.jl` defines `@dxi`, `@dx`, `@dy`, `@dz` so a kernel reads spacing the same
way whether the grid is uniform or not:

```julia
dx, dy = @dxi(di, i, j)   # NTuple of Numbers -> constants; of AbstractVectors -> di[1][i], di[2][j]
_dy    = inv(@dy(di_center, j))
```

They expand to `Base.@propagate_inbounds` accessors, so a nonuniform grid costs one index
and a uniform grid costs nothing. Write new kernels against these rather than closing over
a scalar `dx`, otherwise the kernel silently breaks on refined meshes.

## Coordinate helpers

`x_g(idx, dxi, nxi_or_A)`, `y_g`, `z_g` map a local index to a global coordinate, honoring
the MPI decomposition and periodicity. They are JustRelax's variants of the
ImplicitGlobalGrid functions and are exported from the root module.

`lazy_grid(di, ni, Li; origin)` and `lazy_grid_MPI(di, ni; origin)` build the `xci`/`xvi`
`LinRange`s; `velocity_grids(xci, xvi, di)` builds `xi_vel`. `legacy_uniform_grid(ni, di)`
reconstructs a `Geometry` from spacing alone and is what the `di`-instead-of-`grid` solver
methods call.

## Conventions when writing grid code

- `ni` always means cells. Vertices are `ni .+ 1`. Do not infer one from an array size
  without checking which staggered location the array belongs to.
- Iterate `eachindex`/`axes`, or `@idx ni` for `@parallel` launches; do not hard-code
  `1:size(A, d)`.
- The vertical direction is the **last** dimension and points upwards, so the last entry of
  a column is the shallowest cell — `compute_lithostatic_pressure!` relies on this.
- A function annotated `AbstractArray` promises to work for views, `OffsetArray`s, and
  device arrays. If it genuinely needs 1-based indexing, declare it with
  `Base.require_one_based_indexing`.
