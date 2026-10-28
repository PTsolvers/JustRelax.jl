# Run with a Julia environment containing PoissonGrids 0.1.1:
# julia --project=. --startup-file=no Rift/benchmark_grid_1d.jl > /tmp/grid_bench.csv
# Optional arguments: comma-separated cell counts, iteration limits, repetitions, tolerance.
# Use `adaptive` as the second argument to benchmark grid_solver_options(nc).
using PoissonGrids
using Logging
using Printf
using Statistics

include(joinpath(@__DIR__, "grid_solver_options.jl"))

mutable struct GridBenchmarkLogger <: AbstractLogger
    iterations::Int
    residual::Float64
    nc::Int
end

Logging.min_enabled_level(::GridBenchmarkLogger) = Logging.Info
Logging.shouldlog(::GridBenchmarkLogger, args...) = true
Logging.catch_exceptions(::GridBenchmarkLogger) = false
function Logging.handle_message(logger::GridBenchmarkLogger, level, message, args...; kwargs...)
    matched = match(r"iter\. (\d+) --- abs\. \|r\| =\s+(\S+)", string(message))
    if matched !== nothing
        logger.iterations = parse(Int, matched[1])
        logger.residual = parse(Float64, matched[2]) / logger.nc^0.25
    end
    return nothing
end

function benchmark_grid_1d(
        resolutions = [256, 450, 451, 512, 1024, 2048, 4096, 8192],
        limits = [20_000, 200_000], repetitions = 1, tol = 1.0e-7; adaptive = false
    )
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    monitor = window_monitor(5.0, 0.25, 32.5, -27.5)
    println(stderr, "Julia $VERSION, PoissonGrids $(pkgversion(PoissonGrids)), $(Sys.ARCH), $(Threads.nthreads()) Julia thread(s)")
    settings = adaptive ? "adaptive options per resolution" : "tol=$tol"
    println(stderr, "Domain [-200, 10] km; window_monitor(5, 0.25, 32.5, -27.5); $settings")
    # Compile both diagnostic logging and silent timing paths before measurement.
    with_logger(GridBenchmarkLogger(0, Inf, 32)) do
        solve_grid(-200.0, 10.0, monitor, 32; verbose = true, maxiter = 200_000)
    end
    solve_grid(-200.0, 10.0, monitor, 32; maxiter = 200_000)
    println("nc,maxiter,tolerance,converged,iterations,residual,median_seconds,median_bytes,min_dz_km,max_dz_km")
    for nc in resolutions, budget in limits
        options = adaptive ? grid_solver_options(nc) : (; tol, maxiter = budget)
        println(stderr, "Running nc=$nc, options=$options")
        flush(stderr)
        logger = GridBenchmarkLogger(0, Inf, nc)
        vertices = nothing
        try
            vertices = with_logger(logger) do
                solve_grid(-200.0, 10.0, monitor, nc; verbose = true, options...)
            end
        catch err
            err isa ConvergenceError || rethrow()
            logger.iterations = err.maxiter
            logger.residual = err.residual
        end
        times, bytes = Float64[], Int[]
        for repetition in 1:repetitions
            GC.gc()
            measured = @timed try
                solve_grid(-200.0, 10.0, monitor, nc; options...)
            catch err
                err isa ConvergenceError || rethrow()
                nothing
            end
            push!(times, measured.time)
            push!(bytes, measured.bytes)
        end
        dzmin, dzmax = if vertices === nothing
            NaN, NaN
        else
            @assert length(vertices) == nc + 1
            @assert first(vertices) == -200.0 && last(vertices) == 10.0
            @assert all(isfinite, vertices) && all(>(0), diff(vertices))
            extrema(diff(vertices))
        end
        @printf(
            "%d,%d,%.9e,%s,%d,%.9e,%.6f,%.0f,%.9e,%.9e\n",
            nc, options.maxiter, options.tol, vertices !== nothing, logger.iterations, logger.residual,
            median(times), median(bytes), dzmin, dzmax
        )
        flush(stdout)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    resolutions = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ',')) : [256, 450, 451, 512, 1024, 2048, 4096, 8192]
    adaptive = length(ARGS) >= 2 && ARGS[2] == "adaptive"
    limits = adaptive ? [0] : length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ',')) : [20_000, 200_000]
    repetitions = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    tol = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0e-7
    benchmark_grid_1d(resolutions, limits, repetitions, tol; adaptive)
end
