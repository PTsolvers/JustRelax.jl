# if input is a structure, take the innermost name
# i.e. stokes.V.Vx => "Vx"
macro namevar(x)
    name = split(string(x), ".")[end]
    return quote
        tmp = $(esc(x))
        if tmp isa Number
            $(esc(name)), tmp
        else
            $(esc(name)), Array(tmp)
        end
    end
end

macro namevar(x, T)
    name = split(string(x), ".")[end]
    return quote
        tmp = $(esc(x))
        type = $(esc(T))
        if tmp isa Number
            $(esc(name)), type(tmp)
        else
            $(esc(name)), type.(Array(tmp))
        end
    end
end

"""
    checkpointing_hdf5(dst, stokes, T, time, timestep; precision = Float32)

Save the state of the model at `time` in `dst` as the HDF5 file `checkpoint.h5`, so that the
run can be restarted from it with [`load_checkpoint_hdf5`](@ref).

The file holds the velocity components of `stokes` (`Vx`, `Vy`, and `Vz` in 3D), its
pressure and viscosity, the temperature `T`, and `time`/`timestep`. Fields are transferred
to the CPU before writing, whatever the backend. The file is written to a temporary
directory first and moved into place, so an interrupted call leaves any previous checkpoint
intact.

# Arguments
- `dst`: Directory the checkpoint is written to; created if it does not exist.
- `stokes`: `JustRelax.StokesArrays` holding the velocity, pressure, and viscosity fields.
- `T`: Temperature field.
- `time`: Simulation time.
- `timestep`: Time step.

# Keyword arguments
- `precision`: element type the arrays are converted to before writing.
"""
function checkpointing_hdf5(dst, stokes, T, time, timestep; precision = Float32)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist
    fname = joinpath(dst, "checkpoint")

    # @namevar(time)

    # Create a temporary directory
    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))
        h5open("$(tmpfname).h5", "w") do file
            write(file, @namevar(time)...)
            write(file, @namevar(timestep)...)
            write(file, @namevar(stokes.V.Vx, precision)...)
            write(file, @namevar(stokes.V.Vy, precision)...)
            if !isnothing(stokes.V.Vz)
                write(file, @namevar(stokes.V.Vz, precision)...)
            end
            write(file, @namevar(stokes.P, precision)...)
            write(file, @namevar(stokes.viscosity.η, precision)...)
            return write(file, @namevar(T, precision)...)
        end
        # Move the checkpoint file from the temporary directory to the destination directory
        return mv("$(tmpfname).h5", "$(fname).h5"; force = true)
    end

    return nothing
end

"""
    load_checkpoint_hdf5(file_path)

Load the state of the simulation from an .h5 file.

# Arguments
- `file_path`: The path to the .h5 file.

# Returns
- `P`: The loaded state of the pressure variable.
- `T`: The loaded state of the temperature variable.
- `Vx`: The loaded state of the x-component of the velocity variable.
- `Vy`: The loaded state of the y-component of the velocity variable.
- `Vz`: The loaded state of the z-component of the velocity variable, or `nothing` for a
  2D checkpoint.
- `η`: The loaded state of the viscosity variable.
- `t`: The loaded simulation time.
- `dt`: The loaded time step.

All arrays are returned on the CPU, at the `precision` they were written with; move them to
the device with `PTArray(backend)(A)` before copying them back into a `StokesArrays`. See
[`checkpointing_hdf5`](@ref) for the writing side.

# Example
```julia
file_path = joinpath("path/to/your/output", "checkpoint.h5")
P, T, Vx, Vy, Vz, η, t, dt = load_checkpoint_hdf5(file_path)
```
"""
function load_checkpoint_hdf5(file_path)
    h5file = h5open(file_path, "r")  # Open the file in read mode
    P = read(h5file["P"])  # Read the stokes variable
    T = read(h5file["T"])  # Read the thermal.T variable
    Vx = read(h5file["Vx"])  # Read the stokes.V.Vx variable
    Vy = read(h5file["Vy"])  # Read the stokes.V.Vy variable
    if "Vz" in keys(h5file)  # Check if the "Vz" key exists
        Vz = read(h5file["Vz"])  # Read the stokes.V.Vz variable
    else
        Vz = nothing  # Assign a default value to Vz
    end
    η = read(h5file["η"])  # Read the stokes.viscosity.η variable
    t = read(h5file["time"])  # Read the t variable
    dt = read(h5file["timestep"])  # Read the t variable
    close(h5file)  # Close the file
    return P, T, Vx, Vy, Vz, η, t, dt
end

"""
    save_hdf5(dst, fname, data...)

Save each entry of `data` as the `fname.h5` HDF5 file in the folder `dst`, creating the
folder if it does not exist.
"""
function save_hdf5(dst, fname, data::Vararg{Any, N}) where {N}
    !isdir(dst) && mkpath(dst) # creat folder in case it does not exist
    pth_name = joinpath(dst, fname)
    return save_hdf5(pth_name, data)
end

# comm_cart, info comm_cart, MPI.Info()
function save_hdf5(fname, dim_g, I, comm_cart, info, data::Vararg{Any, N}) where {N}
    @assert HDF5.has_parallel()
    h5open("$(fname).h5", "w", comm_cart, info) do file
        for data_i in data
            name, field = @namevar data_i
            dset = create_dataset(
                file, "/" * name, datatype(eltype(field)), dataspace(dim_g)
            )
            dset[I.indices...] = Array(field)
        end
    end
    return nothing
end

"""
    save_hdf5(fname, data...; precision = Float32)

Save each entry of `data` as the `fname.h5` HDF5 file, one variable per entry, named after
the variable passed at the call site.

# Keyword arguments
- `precision`: element type the arrays are converted to before writing.
"""
function save_hdf5(fname, data::Vararg{Any, N}; precision = Float32) where {N}
    return h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i, precision)
        end
    end
end

"""
    save_data(file, data, precision)
    save_data(file, grid::Geometry)

Write `data` (converted to `precision`) into the open HDF5 `file` under its own variable
name. The `Geometry` method instead writes the cell-center/vertex coordinate vectors
(`Xc`/`Yc`[/`Zc`], `Xv`/`Yv`[/`Zv`]). Used internally by [`save_hdf5`](@ref).
"""
@inline save_data(file, data, precision) = write(file, @namevar(data, precision)...)

function save_data(file, data::Geometry{N}) where {N}
    xci = center_coordinates(data)
    xvi = vertex_coordinates(data)

    write(file, "Xc", xci[1])
    write(file, "Yc", xci[2])
    write(file, "Xv", xvi[1])
    write(file, "Yv", xvi[2])
    if N == 3
        write(file, "Zc", xci[3])
        write(file, "Zv", xvi[3])
    end

    return nothing
end

"""
    center_coordinates(grid::Geometry)

The cell-center coordinate vectors of `grid` (`grid.xci`), collected into plain `Vector`s
for serialization.
"""
center_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xci[i]), Val(N))

"""
    vertex_coordinates(grid::Geometry)

The cell-vertex coordinate vectors of `grid` (`grid.xvi`), collected into plain `Vector`s
for serialization.
"""
vertex_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xvi[i]), Val(N))
