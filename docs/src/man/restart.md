# Restarting from a Checkpoint File

To restart a simulation from a previously saved checkpoint file, you can make use of the checkpointing functions described in the [Checkpointing documentation](./checkpointing.md). Depending on the file format you used to save your checkpoint (HDF5 or JLD2), you can load the saved state of your simulation using the corresponding loading function.

In this example, we will demonstrate how to set up a script to restart a simulation from a JLD2 checkpoint file as we can save the entire structurs of the `StokesArrays` and `ThermalArrays` which makes it easier to restart the simulation. We will assume that you have already saved a checkpoint file using the `checkpointing_jld2` function for the example of a 2D subduction model.
Ideally, one does not need to change much in the initial script used to start the simulation from scratch. The main difference is that instead of initializing the `StokesArrays` and `ThermalArrays` from scratch, we will load them from the checkpoint file.
For a detailed description of the 2D subduction model setup, please refer to the [2D subduction documentation](./subduction2D/subduction2D.md). The following example can be found [here](https://github.com/PTsolvers/JustRelax.jl/blob/d63ca8f08860859700913b575c9befc33d5c4f2a/miniapps/subduction/2D/Subduction2D_restart).

Load JustRelax necessary modules and define backend.
```julia
using CUDA # comment this out if you are not using CUDA; or load AMDGPU.jl if you are using an AMD GPU
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CUDABackend  # Options: CPUBackend, CUDABackend, AMDGPUBackend
```

For this benchmark we will use particles to track the advection of the material phases and their information. For this, we will use [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl)
```julia
using JustPIC, JustPIC._2D
const backend = CUDABackend # Options: JustPIC.CPUBackend, CUDABackend, JustPIC.AMDGPUBackend
```

!!! tip "Script" Leave most of your original script unchanged and only change the parts we highlight in this example, unless you want to explicitly change some model parameters (e.g., rheology, boundary conditions, etc.). Make sure you dont accidentally overwrite your loaded arrays/particles with new initializations.

## Load and initialize particles fields
The `JustPIC` specific function `TA()` will convert the loaded particles to the correct backend.
```julia
data = load(joinpath("Your_checkpointing_directory", "particles.jld2"))
particles     = TA(backend)(Float64, data["particles"])
phases        = TA(backend)(Float64, data["phases"])
phase_ratios  = TA(backend)(Float64, data["phase_ratios"])
particle_args = TA(backend).(Float64, data["particle_args"])
subgrid_arrays  = SubgridDiffusionCellArrays(particles)
# velocity staggered grids
grid_vxi        = velocity_grids(xci, xvi, di)
```

## Load Stokes and Thermal arrays from checkpoint file
:::code-group
```julia [Normal use]
dst = "Your_checkpointing_directory"
stokes_cpu, thermal_cpu, t, dt = load_checkpoint_jld2(dst)
```
```julia [MPI]
dst = "Your_checkpointing_directory"
stokes_cpu, thermal_cpu, t, dt = load_checkpoint_jld2(dst, igg)
```
```julia [Additional fields]
dst = "Your_checkpointing_directory"
fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
stokes_cpu, thermal_cpu, t, dt, it, custom_field_1, custom_field_2 = JLD2.load(fname)
```
:::

The loaded arrays are CPU arrays, so we need to convert them to the correct backend.
```julia
stokes = PTArray(backend_JR, stokes_cpu)
thermal = PTArray(backend_JR, thermal_cpu)
```
From here on you should be able to continue the simulation as usual. Make sure to adjust the time loop to start from the loaded time `t` and iteration `it` if you loaded them.
