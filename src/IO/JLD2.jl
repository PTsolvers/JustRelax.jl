checkpoint_name(dst) = "$dst/checkpoint.jld2"
checkpoint_name(dst, igg::IGG) = "$dst/checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2"

"""
    checkpointing_jld2(dst, stokes, [thermal,] time, timestep[, igg]; kwargs...)

Save Stokes and optional thermal arrays in the directory `dst` as CPU arrays.
The serial method writes `checkpoint.jld2`; passing `igg` writes a rank-specific
file such as `checkpoint0000.jld2`. Each call replaces the previous file at that
path. Restart MPI checkpoints with the same domain decomposition.

Load the checkpoint with [`load_checkpoint_jld2`](@ref), or use `JLD2.load` on
the file to access additional fields saved through keyword arguments.

# Arguments
- `dst`: The destination directory where the checkpoint file will be saved.
- `stokes`: The stokes flow variables to be saved.
- `thermal`: (Optional) The thermal variables to be saved.
- `time`: The current simulation time.
- `timestep`: The current timestep.
- `igg`: (Optional) The IGG struct for parallel runs.

## Keyword Arguments
- `kwargs...`: Additional variables to be saved in the checkpoint file. These will be added to the base checkpoint data.

   # Example
    ```julia
    checkpointing_jld2(
        "path/to/dst",
        stokes,
        thermal,
        t,
        dt,
        igg;
        it = 500,
        example_vec = example_vector,
        additional_data = some_data,
    )

    ```
"""
function checkpointing_jld2(dst, stokes, thermal, time, timestep; kwargs...)
    fname = checkpoint_name(dst)
    checkpointing_jld2(dst, stokes, thermal, time, timestep, fname; kwargs...)
    return nothing
end

function checkpointing_jld2(dst, stokes, thermal, time, timestep, igg::IGG; kwargs...)
    fname = checkpoint_name(dst, igg)
    checkpointing_jld2(dst, stokes, thermal, time, timestep, fname; kwargs...)
    return nothing
end

function checkpointing_jld2(dst, stokes, time, timestep; kwargs...)
    fname = checkpoint_name(dst)
    checkpointing_jld2(dst, stokes, nothing, time, timestep, fname; kwargs...)
    return nothing
end

function checkpointing_jld2(dst, stokes, time, timestep, igg::IGG; kwargs...)
    fname = checkpoint_name(dst, igg)
    checkpointing_jld2(dst, stokes, nothing, time, timestep, fname; kwargs...)
    return nothing
end

function checkpointing_jld2(dst, stokes, thermal, time, timestep, fname::String; kwargs...)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist

    # Create a temporary directory
    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))

        # Build args dict dynamically
        args = Dict(
            :stokes => Array(stokes),
            :time => time,
            :timestep => timestep,
        )

        # Only add thermal if it's not nothing
        if !isnothing(thermal)
            args[:thermal] = Array(thermal)
        end

        # Add any additional kwargs dynamically using their names as keys
        for (key, value) in pairs(kwargs)
            args[key] = isnothing(value) ? nothing :
                isa(value, AbstractArray) ? Array(value) :
                isa(value, Tuple) ? Array.(value) : value
        end
        try
            jldsave(tmpfname; args...)
        catch
            jldsave(tmpfname, IOStream; args...)
        end
        # Move the checkpoint file from the temporary directory to the destination directory
        return mv(tmpfname, fname; force = true)
    end

    return nothing
end
"""
    load_checkpoint_jld2(dst[, igg])

Load a checkpoint from the directory `dst` (not a file path). The serial method
reads `checkpoint.jld2`; passing `igg` selects the file matching its MPI rank.

Return `(stokes, thermal, time, timestep)`, with arrays on the CPU and `thermal`
set to `nothing` if it was not saved. Additional fields are available by loading
the file directly with `JLD2.load`; the saved time-step key is `"timestep"`.

# Example
```julia
stokes, thermal, time, timestep = load_checkpoint_jld2("path/to/dst", igg)
```
Or, for a serial checkpoint without thermal arrays:
```julia
stokes, _, time, timestep = load_checkpoint_jld2("path/to/dst")
```
"""
function load_checkpoint_jld2(file_path)
    fname = checkpoint_name(file_path)
    restart = load(fname)  # Load the file
    stokes = restart["stokes"]  # Read the stokes variable
    thermal = haskey(restart, "thermal") ? restart["thermal"] : nothing  # Read thermal if present
    time = restart["time"]  # Read the time variable
    timestep = restart["timestep"]  # Read the timestep variable
    return stokes, thermal, time, timestep
end

function load_checkpoint_jld2(file_path, igg::IGG)
    fname = checkpoint_name(file_path, igg)
    restart = load(fname)  # Load the file
    stokes = restart["stokes"]  # Read the stokes variable
    thermal = haskey(restart, "thermal") ? restart["thermal"] : nothing  # Read thermal if present
    time = restart["time"]  # Read the time variable
    timestep = restart["timestep"]  # Read the timestep variable
    return stokes, thermal, time, timestep
end
