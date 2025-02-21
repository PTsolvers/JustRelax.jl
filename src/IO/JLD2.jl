"""
    checkpointing_jld2(dst, stokes, thermal, time, timestep, igg)

Save necessary data in `dst` as a jld2 file to restart the model from the state at `time`.
If run in parallel, the file will be named after the corresponidng rank e.g. `checkpoint0000.jld2`
and thus can be loaded by the processor while restarting the simulation.
If you want to restart your simulation from the checkpoint you can use load() and specify the MPI rank
by providing a dollar sign and the rank number.

   # Example
    ```julia
    checkpointing_jld2(
        "path/to/dst",
        stokes,
        thermal,
        t,
        igg,
    )

    ```
"""
checkpoint_name(dst) = "$dst/checkpoint.jld2"
checkpoint_name(dst, igg::IGG) = "$dst/checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2"

function checkpointing_jld2(dst, stokes, thermal, time, timestep)
    fname = checkpoint_name(dst)
    checkpointing_jld2(dst, stokes, thermal, time, timestep, fname)
    return nothing
end

function checkpointing_jld2(dst, stokes, thermal, time, timestep, igg::IGG)
    fname = checkpoint_name(dst, igg)
    checkpointing_jld2(dst, stokes, thermal, time, timestep, fname)
    return nothing
end

function checkpointing_jld2(dst, stokes, thermal, time, timestep, fname::String)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist

    # Create a temporary directory
    mktempdir() do tmpdir
        # Save the checkpoint file in the temporary directory
        tmpfname = joinpath(tmpdir, basename(fname))
        jldsave(
            tmpfname;
            stokes = Array(stokes),
            thermal = Array(thermal),
            time = time,
            timestep = timestep,
        )
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
- `thermal`: The loaded state of the thermal variable.
- `time`: The loaded simulation time.
- `timestep`: The loaded time step.
"""
function load_checkpoint_jld2(file_path)
    restart = load(file_path)  # Load the file
    stokes = restart["stokes"]  # Read the stokes variable
    thermal = restart["thermal"]  # Read the thermal variable
    time = restart["time"]  # Read the time variable
    timestep = restart["timestep"]  # Read the timestep variable
    return stokes, thermal, time, timestep
end
