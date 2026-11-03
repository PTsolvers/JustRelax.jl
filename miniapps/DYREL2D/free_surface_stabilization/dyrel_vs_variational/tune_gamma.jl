# Sweep the Powell-Hestenes penalty scaling `γfact` for both DYREL Rayleigh-Taylor runs and
# report, per solver, the value that costs the fewest dynamic-relaxation iterations.
#
#   DYREL_CUDA=true DYREL_N=64 DYREL_TUNE_STEPS=15 \
#   julia --project=<thisdir> <thisdir>/tune_gamma.jl
#
# One process handles one resolution: `@init_parallel_stencil` and `init_global_grid` are both
# once-per-session, so sweeping γ inside the session pays compilation once rather than once per
# candidate.
#
# Each finished candidate is appended to `gamma_sweep.csv` before the next one starts, and any
# candidate already in that file is skipped. A γ that drives the model unstable makes the marker
# chain leave the grid and faults on the device, which poisons the CUDA context and takes the
# process down — so the run is designed to be relaunched (see `drive_tune.sh`), resuming from
# the rows already written. `gamma_sweep.inflight` names the candidate being run, so the driver
# can record the one that died.
#
# A γ is ranked only if it completed every step and converged on all of them: an unconverged
# step stops at `total_iterMax`, so its cost is censored and would otherwise look cheap.

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "config.jl"))

@static if IS_CUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D

const backend_JR = @static if IS_CUDA
    CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if IS_CUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D

const backend_JP = @static if IS_CUDA
    CUDABackend
else
    JustPIC.CPUBackend
end

using GeoParams, Random, Printf, Statistics, DelimitedFiles

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "runners.jl"))

const TUNE_STEPS = parse(Int, get(ENV, "DYREL_TUNE_STEPS", "15"))
const GAMMAS = parse.(Float64, split(get(ENV, "DYREL_GAMMAS", "20,35,50,75,100,150,200,300"), ','))
const SOLVERS = string.(split(get(ENV, "DYREL_TUNE_SOLVERS", "standard,variational"), ','))
const TUNE_OUTDIR = joinpath(@__DIR__, get(ENV, "DYREL_TUNE_OUTDIR", "tune_gamma_n$(N)"))
const CSV = joinpath(TUNE_OUTDIR, "gamma_sweep.csv")
const INFLIGHT = joinpath(TUNE_OUTDIR, "gamma_sweep.inflight")

const HEADER = ["solver", "N", "gamma_fact", "steps_done", "total_iter", "mean_iter", "median_iter", "max_iter", "unconverged", "seconds", "status"]

runner_for(label) = label == "standard" ? run_standard : run_variational

reclaim!() = (GC.gc(true); @static IS_CUDA ? CUDA.reclaim() : nothing)

"""
    done_candidates()

`(solver, γfact)` pairs already recorded in the CSV, so a relaunch resumes instead of repeating.
"""
function done_candidates()
    isfile(CSV) || return Set{Tuple{String, Float64}}()
    rows, _ = readdlm(CSV, ',', header = true)
    return Set((string(rows[i, 1]), Float64(rows[i, 3])) for i in axes(rows, 1))
end

function append_row!(row)
    new = !isfile(CSV)
    # `permutedims` of a length-11 vector gives the 1×11 matrix `writedlm` needs; a multiline
    # array literal would vcat into several rows instead.
    vals = Any[
        row.solver, row.N, row.γfact, row.steps_done, row.total_iter, row.mean_iter,
        row.median_iter, row.max_iter, row.unconverged, row.seconds, row.status,
    ]
    return open(CSV, "a") do io
        new && writedlm(io, permutedims(HEADER), ',')
        writedlm(io, permutedims(vals), ',')
    end
end

function run_candidate(label, γ, igg)
    @printf("\n=== %s  N=%d  γfact=%.4g  (%d steps) ===\n", label, N, γ, TUNE_STEPS)
    flush(stdout)
    write(INFLIGHT, "$label,$γ")
    reclaim!()

    records = runner_for(label)(igg; nsteps = TUNE_STEPS, γfact = γ)

    iters = getproperty.(records, :iter)
    secs = getproperty.(records, :seconds)
    nbad = count(!, getproperty.(records, :converged))
    row = (;
        solver = label, N, γfact = γ,
        steps_done = length(records),
        total_iter = sum(iters),
        mean_iter = mean(iters),
        median_iter = median(iters),
        max_iter = maximum(iters),
        unconverged = nbad,
        seconds = sum(secs),
        status = nbad == 0 ? "ok" : "unconverged",
    )
    append_row!(row)
    rm(INFLIGHT, force = true)

    @printf(
        "--> %s γ=%.4g: total=%d  mean/step=%.0f  max/step=%d  unconverged=%d  %.1f s\n",
        label, γ, row.total_iter, row.mean_iter, row.max_iter, nbad, row.seconds
    )
    flush(stdout)
    return row
end

function main()
    mkpath(TUNE_OUTDIR)
    done = done_candidates()
    todo = [(s, γ) for s in SOLVERS for γ in GAMMAS if (s, γ) ∉ done]

    if isempty(todo)
        println("all candidates already recorded in $CSV")
        return
    end
    @printf("%d candidate(s) to run, %d already recorded\n", length(todo), length(done))

    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(N, N, 1; init_MPI = true)...)
    else
        IGG(init_global_grid(N, N, 1; init_MPI = false)...)
    end

    for (label, γ) in todo
        run_candidate(label, γ, igg)
    end
    return println("\nsweep complete: $CSV")
end

main()
