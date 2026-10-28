using JLD2

# Copy into the allocated backend containers to preserve particle-field aliases.
function copy_zwaan_state!(dst, src)
    if dst isa JustPIC.CellArray
        size(dst) == size(src) || throw(DimensionMismatch("Checkpoint particle storage differs"))
        copyto!(dst.data, PTArray(backend)(src).data)
    elseif dst isa AbstractArray
        size(dst) == size(src) || throw(DimensionMismatch("Checkpoint field size differs"))
        copyto!(dst, src)
    elseif dst isa Union{Tuple, NamedTuple}
        length(dst) == length(src) || throw(DimensionMismatch("Checkpoint tuple length differs"))
        foreach(copy_zwaan_state!, dst, src)
    elseif dst isa Number || dst isa Symbol || dst isa Type || dst === nothing
        isequal(dst, src) || throw(ArgumentError("Checkpoint metadata differs: $dst versus $src"))
    else
        for f in fieldnames(typeof(dst))
            isdefined(dst, f) || continue
            isdefined(src, f) || throw(ArgumentError("Checkpoint is missing field $f"))
            copy_zwaan_state!(getfield(dst, f), getfield(src, f))
        end
    end
    return nothing
end

"""
    reload_zwaan_checkpoint!(figdir, stokes, thermal, particles, particle_args, phase_ratios, chain, pτ)

Restore the matching serial checkpoint pair from `figdir/checkpoint` (or pass the
checkpoint directory itself). Keep the original grid, rheology and particle capacities.
Returns nondimensional `(time, timestep, iteration)`; arrays stay on the active backend.
"""
function reload_zwaan_checkpoint!(figdir, stokes, thermal, particles, particle_args, phase_ratios, chain, pτ; P_lith = nothing)
    dst = isfile(joinpath(figdir, "checkpoint.jld2")) ? figdir : joinpath(figdir, "checkpoint")
    files = (joinpath(dst, "checkpoint.jld2"), joinpath(dst, "particles_checkpoint.jld2"))
    for file in files
        isfile(file) || throw(ArgumentError("Missing restart file: $file"))
    end
    ck, pk = JLD2.load.(files)
    for (a, b) in (("time", "time"), ("timestep", "timestep"), ("it", "it"))
        isequal(ck[a], pk[b]) || throw(ArgumentError("Checkpoint pair has mismatched $a"))
    end
    for (current, saved) in zip(particles.xvi, pk["particles"].xvi)
        Array(current) == Array(saved) || throw(ArgumentError("Checkpoint grid differs from this run"))
    end
    copy_zwaan_state!(stokes, ck["stokes"])
    copy_zwaan_state!(thermal, ck["thermal"])
    copy_zwaan_state!(particles, pk["particles"])
    copy_zwaan_state!(particle_args, pk["particle_args"])
    copy_zwaan_state!(phase_ratios, pk["phase_ratios"])
    copy_zwaan_state!(chain, pk["chain"])
    # Older checkpoints omitted the FLIP reference, which equals the last τ_o.
    reference = get(pk, "stress_reference", (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy))
    copy_zwaan_state!(pτ.τ_grid, reference)
    if !isnothing(P_lith)
        if haskey(ck, "P_lith")
            copy_zwaan_state!(P_lith, ck["P_lith"])
        else
            P_lith .= stokes.P
            @warn "Checkpoint omitted P_lith; pressure plots use the restart pressure as reference"
        end
    end
    return ck["time"], ck["timestep"], ck["it"]
end
