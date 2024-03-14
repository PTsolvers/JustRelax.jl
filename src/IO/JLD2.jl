"""
    checkpointing(dst, stokes, thermal, η, particles, phases, time)

Save necessary data in `dst` as and jld2 file to restart the model from the state at `time`.
If run in parallel, the file will be named after the corresponidng rank e.g.`checkpoint_rank_0.jld2`
and thus can be loaded by the processor while restarting the simulation.
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
