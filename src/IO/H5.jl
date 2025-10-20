# if input is a structure, take the innermost name
# i.e. stokes.V.Vx => "Vx"
macro namevar(x)
    name = split(string(x), ".")[end]
    return quote
        tmp = $(esc(x))
        if tmp <: Number
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
        if tmp <: Number
            $(esc(name)), T(tmp)
        else
            $(esc(name)), T.(Array(tmp))
        end
    end
end

"""
    checkpointing_hdf5(dst, stokes, T, η, time, timestep)

Save necessary data in `dst` as and HDF5 file to restart the model from the state at `time`
"""
function checkpointing_hdf5(dst, stokes, T, time, timestep; precision = Float32)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist
    fname = joinpath(dst, "checkpoint")

    # Create a temporary directory
    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))
        h5open("$(tmpfname).h5", "w") do file
            write(file, @namevar(time, precision)...)
            write(file, @namevar(timestep, precision)...)
            write(file, @namevar(stokes.V.Vx,precision)...)
            write(file, @namevar(stokes.V.Vy,precision)...)
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
- `Vz`: The loaded state of the z-component of the velocity variable.
- `η`: The loaded state of the viscosity variable.
- `t`: The loaded simulation time.
- `dt`: The loaded simulation time.

# Example
```julia
# Define the path to the .h5 file
file_path = "path/to/your/file.h5"

# Use the load_checkpoint function to load the variables from the file
P, T, Vx, Vy, Vz, η, t, dt = `load_checkpoint(file_path)``

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
    function save_hdf5(dst, fname, data)

Save `data` as the `fname.h5` HDF5 file in the folder `dst`
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
    function save_hdf5(fname, data)

Save `data` as the `fname.h5` HDF5 file
"""
function save_hdf5(fname, data::Vararg{Any, N}; precision=Float32) where {N}
    return h5open("$(fname).h5", "w") do file
        for data_i in data
            save_data(file, data_i, precision)
        end
    end
end

@inline save_data(file, data, precision) = write(file, @namevar(data, precision)...)

function save_data(file, data::Geometry{N}, ::Any) where {N}
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

center_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xci[i]), Val(N))
vertex_coordinates(data::Geometry{N}) where {N} = ntuple(i -> collect(data.xvi[i]), Val(N))
