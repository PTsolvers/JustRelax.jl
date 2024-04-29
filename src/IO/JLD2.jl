"""
    checkpointing_jld2(dst, stokes, thermal, particles, phases, time)

Save necessary data in `dst` as and jld2 file to restart the model from the state at `time`.
If run in parallel, the file will be named after the corresponidng rank e.g. `checkpoint_rank_0.jld2`
and thus can be loaded by the processor while restarting the simulation.
If you want to restart your simulation from the checkpoint you can use load() and specify the MPI rank
by providing a dollar sign and the rank number.

   # Example
    ```julia
    checkpointing_jld2(
        "path/to/dst",
        stokes,
        thermal,
        particles,
        pPhases,
        t,
    )

    restart = load("path/to/dst/checkpoint_rank_(igg.me).jld2")"
    stokes = restart["stokes"]
    thermal = restart["thermal"]
    particles = restart["particles"]
    pPhases = restart["phases"]
    t = restart["time"]
    ```
"""
function checkpointing_jld2(dst, stokes, thermal, particles, phases, time; igg=igg::IGG)
    !isdir(dst) && mkpath(dst) # create folder in case it does not exist
    fname = joinpath(dst, "checkpoint_rank_$(igg.me).jld2")
    return jldsave(
        fname;
        stokes=Array(stokes),
        thermal=Array(thermal),
        # η=_tocpu(η), # to be stored in stokes - PR #130
        particles=Array(particles),
        phases=Array(phases),
        time,
    )
end
