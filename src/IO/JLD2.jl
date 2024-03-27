"""
    checkpointing_jld2(dst, stokes, thermal, η, particles, phases, time)

Save necessary data in `dst` as and jld2 file to restart the model from the state at `time`.
If run in parallel, the file will be named after the corresponidng rank e.g.`checkpoint_rank_0.jld2`
and thus can be loaded by the processor while restarting the simulation.
If you want to restart your simulation from the checkpoint you can use load() and specify the MPI rank
by providing a dollar sign and the rank number.

   # Example
    ```julia
    checkpointing_jld2(
        "path/to/dst",
        stokes,
        thermal,
        η,
        particles,
        pPhases,
        t,
    )

    restart = load("path/to/dst/checkpoint_rank_(igg.me).jld2")"
    stokes = restart["stokes"]
    thermal = restart["thermal"]
    η = restart["η"]
    particles = restart["particles"]
    pPhases = restart["phases"]
    t = restart["time"]
    ```
"""
function checkpointing_jld2(dst, stokes, thermal, η, particles, phases, time; igg=igg::IGG)
    !isdir(dst) && mkpath(dst) # creat folder in case it does not exist
    fname = joinpath(dst, "checkpoint_rank_$(igg.me).jld2")
    return jldsave(
        fname;
        stokes=_tocpu(stokes),
        thermal=_tocpu(thermal),
        η=_tocpu(η),
        particles=_tocpu(particles),
        phases=_tocpu(phases),
        time,
    )
end
