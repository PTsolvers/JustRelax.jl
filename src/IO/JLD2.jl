"""
    checkpointing_jld2(dst, stokes, thermal, time, timestep, igg; kwargs...)

Save necessary data in `dst` as a jld2 file to restart the model from the state at `time`.
If run in parallel, the file will be named after the corresponidng rank e.g. `checkpoint0000.jld2`
and thus can be loaded by the processor while restarting the simulation.
If you want to restart your simulation from the checkpoint you can use load() and specify the MPI rank
by providing a dollar sign and the rank number.

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
checkpoint_name(dst) = "$dst/checkpoint.jld2"
checkpoint_name(dst, igg::IGG) = "$dst/checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2"

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
    load_checkpoint_jld2(file_path)

Load the state of the simulation from a .jld2 file.

# Arguments
- `file_path`: The path to the .jld2 file.

# Returns
- `stokes`: The loaded state of the stokes variable.
- `thermal`: The loaded state of the thermal variable. Can be `nothing` if not present in the file.
- `time`: The loaded simulation time.
- `timestep`: The loaded time step.
- `igg`: The IGG struct if needed for parallel runs.

## Example
```julia
stokes, thermal, time, timestep = load_checkpoint_jld2("path/to/checkpoint.jld2", igg)
```
or without thermal
```julia
stokes, _, time, timestep = load_checkpoint_jld2("path/to/checkpoint.jld2")
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
