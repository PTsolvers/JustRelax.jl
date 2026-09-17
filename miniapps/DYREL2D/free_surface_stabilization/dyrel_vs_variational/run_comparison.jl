# Compare the per-time-step dynamic-relaxation iteration count of the two DYREL Rayleigh-Taylor
# miniapps. Run from anywhere:
#
#   julia --project=miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational \
#         miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational/run_comparison.jl [nsteps]
#
# Both runs share one session, so they share the single permitted `@init_parallel_stencil` call
# and therefore the same backend.

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

using GeoParams, CairoMakie, Random, Printf, Statistics, DelimitedFiles

include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "runners.jl"))

## REPORTING -----------------------------------------------------------------------------

function write_csv(path, standard, variational)
    header = ["step" "dt_kyr" "iter_standard" "err_standard" "converged_standard" "seconds_standard" "iter_variational" "err_variational" "converged_variational" "seconds_variational"]
    rows = mapreduce(vcat, zip(standard, variational)) do (s, v)
        [s.step s.dt_kyr s.iter s.err s.converged s.seconds v.iter v.err v.converged v.seconds]
    end
    return open(path, "w") do io
        writedlm(io, vcat(header, rows), ',')
    end
end

function summarize(io, standard, variational)
    is, iv = getproperty.(standard, :iter), getproperty.(variational, :iter)
    ts, tv = getproperty.(standard, :seconds), getproperty.(variational, :seconds)
    ns, nv = count(!, getproperty.(standard, :converged)), count(!, getproperty.(variational, :converged))

    println(io, "DYREL vs. variational DYREL — Rayleigh-Taylor, $(N)×$(N), $(length(standard)) steps")
    println(io, "backend = $(IS_CUDA ? "CUDA" : "CPU/Threads"), dt = $(ADAPTIVE_DT ? "adaptive" : "$(DT / (1.0e3 * SECYR)) kyr fixed")")
    println(io, "γfact: standard = $GAMMA_FACT_STANDARD, variational = $GAMMA_FACT_VARIATIONAL")
    println(io, "dynamic-relaxation CFL = $DR_CFL, c_fact = $C_FACT, total_iterMax = $TOTAL_ITERMAX")
    println(io)
    @printf(io, "%-24s %14s %14s %10s\n", "", "standard", "variational", "ratio")
    @printf(io, "%-24s %14d %14d %10.2f\n", "total iterations", sum(is), sum(iv), sum(iv) / sum(is))
    @printf(io, "%-24s %14.1f %14.1f %10.2f\n", "mean iterations/step", mean(is), mean(iv), mean(iv) / mean(is))
    @printf(io, "%-24s %14d %14d %10.2f\n", "median iterations/step", round(Int, median(is)), round(Int, median(iv)), median(iv) / median(is))
    @printf(io, "%-24s %14d %14d %10s\n", "min iterations/step", minimum(is), minimum(iv), "")
    @printf(io, "%-24s %14d %14d %10s\n", "max iterations/step", maximum(is), maximum(iv), "")
    @printf(io, "%-24s %14d %14d %10s\n", "unconverged steps", ns, nv, "")
    @printf(io, "%-24s %14.1f %14.1f %10.2f\n", "total wall time [s]", sum(ts), sum(tv), sum(tv) / sum(ts))
    return
end

function plot_comparison(path, standard, variational)
    steps = getproperty.(standard, :step)
    is, iv = getproperty.(standard, :iter), getproperty.(variational, :iter)

    fig = Figure(size = (900, 800))

    ax1 = Axis(
        fig[1, 1], xlabel = "time step", ylabel = "DR iterations",
        title = "Iterations per time step ($(N)×$(N), $(length(steps)) steps)"
    )
    lines!(ax1, steps, is, label = "DYREL")
    lines!(ax1, steps, iv, label = "variational DYREL")
    axislegend(ax1, position = :rt)

    ax2 = Axis(fig[2, 1], xlabel = "time step", ylabel = "cumulative DR iterations")
    lines!(ax2, steps, cumsum(is), label = "DYREL")
    lines!(ax2, steps, cumsum(iv), label = "variational DYREL")
    axislegend(ax2, position = :lt)

    ax3 = Axis(fig[3, 1], xlabel = "time step", ylabel = "iterations, variational / standard")
    lines!(ax3, steps, iv ./ is)
    hlines!(ax3, [1.0], color = :black, linestyle = :dash)

    return save(path, fig)
end

## DRIVER --------------------------------------------------------------------------------

function main(nsteps)
    igg = if !(JustRelax.MPI.Initialized())
        IGG(init_global_grid(N, N, 1; init_MPI = true)...)
    else
        IGG(init_global_grid(N, N, 1; init_MPI = false)...)
    end

    mkpath(OUTDIR)

    @info "running standard DYREL" nsteps
    standard = run_standard(igg; nsteps)

    @info "running variational DYREL" nsteps
    variational = run_variational(igg; nsteps)

    write_csv(joinpath(OUTDIR, "iterations.csv"), standard, variational)
    plot_comparison(joinpath(OUTDIR, "iterations.png"), standard, variational)

    open(joinpath(OUTDIR, "summary.txt"), "w") do io
        summarize(io, standard, variational)
    end
    summarize(stdout, standard, variational)

    @info "wrote results" OUTDIR
    return standard, variational
end

main(isempty(ARGS) ? NSTEPS : parse(Int, ARGS[1]))
