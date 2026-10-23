# Santorini sea-level miniapps

`Santorini_SeaLevel3D.jl` solves the fixed-geometry, viscoelastic chamber model
with 3D variational DYREL. It includes the prescribed seawater load, bulk
compressibility, Maxwell shear elasticity, and free-surface stabilization.
The Gershgorin estimate uses the same phase-dependent Maxwell viscosity at
cell centers and shear edges as the stress operator.

Run from the repository root with the miniapps environment. CUDA is the default:

```sh
julia --project=miniapps --startup-file=no SeaLevel/Santorini_SeaLevel3D.jl
```

For a smaller CPU run without plots:

```sh
JULIA_JUSTRELAX_BACKEND=CPU JULIA_NUM_THREADS=8 SANTORINI_PLOT=false \
NX=48 NY=48 NZ=32 SANTORINI_STEPS=2 \
julia --project=miniapps --startup-file=no SeaLevel/Santorini_SeaLevel3D.jl
```

The script defaults to the synthetic `:steps` loading history. To choose a
history, disable VTK, or choose an output directory, include the file and call
`main` explicitly after initializing `IGG`:

```julia
include("SeaLevel/Santorini_SeaLevel3D.jl")
igg = IGG(init_global_grid(48, 48, 32)...)
out = main(igg, 48, 48, 32;
    history = :steps, max_steps = 2, do_vtk = false, figdir = "/tmp/santorini3d")
@assert all(isfinite, out.σ1_roof)
```

Set `JULIA_JUSTRELAX_BACKEND` and `SANTORINI_PLOT` before including the file.
`main` returns the sampled times and chamber-wall stress changes. Its lateral
lithostatic-pressure average requires one MPI rank along both horizontal axes.
Coarse runs check execution and convergence; resolving the 500 m layers and
1 km chamber thickness requires a finer mesh.
