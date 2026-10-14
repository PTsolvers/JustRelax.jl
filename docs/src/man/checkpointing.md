# Checkpointing

It is common to save the state of a simulation at regular intervals, especially for long-running simulations. This allows you to restart the simulation from the last saved state in case of interruptions or to continue the simulation at a later time without losing progress. JustRelax provides a simple way to save and load checkpoint files. Two checkpointing functions are available for the most common file extensions (HDF5 and JLD2). By loading the `DataIO` module, you gain access to these checkpointing functions as well as VTK saving functions for later visualization with [ParaView](https://www.paraview.org/).
For VTK output, see [Visualization](./visualization.md).

!!! tip "JustPIC checkpointing"
    A similar checkpointing function is defined by [JustPIC.jl](https://juliageodynamics.github.io/JustPIC.jl/dev/IO/) to save the state of the particles.


:::code-group

```julia [2D module]
using JustRelax, JustRelax.JustRelax2D
using JustRelax.DataIO
```

```julia [3D module]
using JustRelax, JustRelax.JustRelax3D
using JustRelax.DataIO
```
:::

Use JLD2 to preserve the complete Stokes and optional thermal containers. HDF5 saves a subset of fields for reconstruction into an existing model.

### Saving and loading checkpoint with HDF5
The HDF5 checkpointing function saves the most important model variables (pressure, temperature, velocity components, viscosity, time, and timestep) to a `checkpoint.h5` file in your destination folder.

```julia
dst = "Your_checkpointing_directory"
checkpointing_hdf5(dst, stokes, thermal.T, time, timestep)
```

To load the checkpoint, use `load_checkpoint_hdf5`. This function returns the following tuple (`Vz` is `nothing` for a 2D checkpoint):

```julia
fname = joinpath(dst, "checkpoint.h5")
P, T, Vx, Vy, Vz, η, t, dt = load_checkpoint_hdf5(fname)
```

HDF5 converts fields to `Float32` by default; pass `precision = Float64` to
`checkpointing_hdf5` when you need double-precision fields.

### Saving and loading checkpoint with JLD2
JLD2 saves all Stokes arrays and, when supplied, the thermal arrays. The serial
method writes `checkpoint.jld2`; passing `igg` writes one file per MPI rank,
such as `checkpoint0000.jld2` and `checkpoint0001.jld2`. Additional keyword
arguments save custom fields.

Each call replaces the previous checkpoint at the same destination. Use a
separate directory for each saved time if you want to retain checkpoint history.
Restart MPI checkpoints with the same domain decomposition. The HDF5 writer has
no rank-aware overload; use a distinct destination directory per rank.

!!! warning "Checkpointing"
    All checkpointing functions save the arrays as CPU arrays no matter the backend. On a GPU backend the arrays are transferred to the CPU before saving, which takes time proportional to the size of the model.

:::code-group

```julia [Normal use]
dst = "Your_checkpointing_directory"
checkpointing_jld2(dst, stokes, thermal, time, dt)
```

```julia [MPI]
dst = "Your_checkpointing_directory"
checkpointing_jld2(dst, stokes, thermal, time, dt, igg)
```

```julia [Additional fields]
dst = "Your_checkpointing_directory"
checkpointing_jld2(dst, stokes, thermal, time, dt, igg; it = it, custom_field_1 = some_data, custom_field_2 = example_vector)
```
:::

Pass the checkpoint **directory**, not the filename, to `load_checkpoint_jld2`. If thermal arrays were omitted when saving, its second return value is `nothing`.

To load the checkpoint, you can use the preexisting `load_checkpoint_jld2` function or use the `JLD2` loading function directly. The `load_checkpoint_jld2` function is MPI agnostic and will automatically load the correct file for each processor based on its rank:

:::code-group

```julia [Normal use]
dst = "Your_checkpointing_directory"
stokes, thermal, t, dt = load_checkpoint_jld2(dst)
```

```julia [MPI]
dst = "Your_checkpointing_directory"
stokes, thermal, t, dt = load_checkpoint_jld2(dst, igg)
```
:::

If you save additional fields, it is the easiest to load the checkpointing file directly using the `JLD2` package. This way, you can access all saved variables by their names:

```julia
using JLD2
fname = joinpath(dst, "checkpoint.jld2") # Serial checkpoint
# For MPI, use instead:
# fname = joinpath(dst, "checkpoint" * lpad(string(igg.me), 4, '0') * ".jld2")
data = JLD2.load(fname)
```
which then returns a dictionary with all your saved variables. You can access them like this:

```julia
stokes = data["stokes"]
thermal = data["thermal"]
t = data["time"]
dt = data["timestep"]
custom_field_1 = data["custom_field_1"]
custom_field_2 = data["custom_field_2"]
# and so on...
```
