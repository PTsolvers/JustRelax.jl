module JustRelaxBenchmarks

using Chairmarks: @be
using Dates: UTC, now
using JSON: JSON
using JustRelax
using JustRelax: JustRelax2D, JustRelax3D
using ImplicitGlobalGrid: finalize_global_grid, init_global_grid
using MPI: MPI
using Statistics: mean, median, quantile

export benchmark_cases, dashboard_main, main, print_comparison, run_benchmarks, write_dashboard_data,
    write_results

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PERFORMANCE_MODEL_VERSION = "justrelax_teff_v2"
const REPOSITORY_URL = "https://github.com/PTsolvers/JustRelax.jl"

# Effective memory access per pseudo-transient iteration, counted as in Räss et al. (2022,
# GMD 15, 5757): each unknown field is read and written once, each known field is read once,
# and staggering is ignored (every field has `nᴰ` entries). Auxiliary fields such as stresses,
# fluxes, and iteration parameters are excluded. FLOPs are counted per cell and iteration from
# the solver kernels; benchmarking/README.md lists the counting rules and coefficients.
struct PerformanceModel
    flops::Int
    memory_bytes::Int
    description::String
end

struct BenchmarkCase{S, R, V}
    name::String
    group::String
    grid_cells::NTuple
    setup::S
    run::R
    validate::V
    work_units::Int
    work_unit::String
    parameters::Dict{String, Any}
    performance_model::PerformanceModel
end

teff_model(flops_per_cell, unknowns, knowns, cells, ::Type{T}, description) where {T} =
    PerformanceModel(flops_per_cell * cells, (2 * unknowns + knowns) * cells * sizeof(T), description)

grid_label(ni) = join(ni, "×")

device_array(backend, A) = JustRelax.PTArray(backend)(A)

function validate_fields(label, fields...)
    for field in fields
        all(isfinite, Array(field)) || error("$label benchmark produced nonfinite values")
    end
    return nothing
end

# `solve!` only reads the residual every `nout` iterations; an `nout` larger than `iterMax`
# skips that check, so the solver runs exactly `iterMax + 1` iterations.
const NO_RESIDUAL_CHECK = 10^9

# Infinite shear and bulk moduli make the visco-elastic solver purely viscous and incompressible;
# both moduli are infinite, so their order in the 2D (G, K) and 3D (K, G) signatures is moot.
elastic_moduli(backend, ni, ::Type{T}) where {T} = ntuple(_ -> device_array(backend, fill(T(Inf), ni)), 2)

"""
Viscous Stokes flow driven by a SolCx-like buoyancy field, free-slip on every side, with a
smooth viscosity contrast.
"""
function stokes_state(backend, ::Val{2}, ni, ::Type{T}) where {T}
    li = (one(T), one(T))
    grid = JustRelax2D.Geometry(ni, li; origin = (zero(T), zero(T)))
    xc, yc = Array.(grid.xci)
    stokes = JustRelax2D.StokesArrays(backend, ni)
    stokes.viscosity.η .= device_array(backend, T[1 + 9 * (x > 0.5) for x in xc, _ in yc])
    ρg = (
        device_array(backend, zeros(T, ni)),
        device_array(backend, T[-sin(π * y) * cos(π * x) for x in xc, y in yc]),
    )
    flow_bcs = JustRelax2D.VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true)
    )
    pt_stokes = JustRelax2D.PTStokesCoeffs(li, grid.di.center; CFL = 1 / √2.1)
    return (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli = elastic_moduli(backend, ni, T), iterations = Ref(0))
end

function stokes_state(backend, ::Val{3}, ni, ::Type{T}) where {T}
    li = (one(T), one(T), one(T))
    grid = JustRelax3D.Geometry(ni, li; origin = (zero(T), zero(T), zero(T)))
    xc, yc, zc = Array.(grid.xci)
    stokes = JustRelax3D.StokesArrays(backend, ni)
    stokes.viscosity.η .= device_array(backend, T[1 + 9 * (x > 0.5) for x in xc, _ in yc, _ in zc])
    ρg = (
        device_array(backend, zeros(T, ni)),
        device_array(backend, zeros(T, ni)),
        device_array(backend, T[-sin(π * z) * cos(π * x) for x in xc, _ in yc, z in zc]),
    )
    flow_bcs = JustRelax3D.VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    pt_stokes = JustRelax3D.PTStokesCoeffs(li, grid.di.center; CFL = 1 / √3.1)
    return (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli = elastic_moduli(backend, ni, T), iterations = Ref(0))
end

function stokes_case(backend, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    setup() = stokes_state(backend, dims, ni, T)
    function run(state)
        (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli) = state
        out = JR.solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, moduli..., one(T), IGG_HANDLE[];
            kwargs = (; iterMax = iterations - 1, nout = NO_RESIDUAL_CHECK, verbose = false),
        )
        state.iterations[] = out.iter
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("Stokes benchmark ran $(state.iterations[]) iterations instead of $iterations")
        return validate_fields("Stokes", JR.unpack_velocity(state.stokes.V)..., state.stokes.P)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
        "rheology" => "linear viscous (G = K = Inf)",
    )
    model = teff_model(
        D == 2 ? 123 : 222, D + 1, 2, n^D * iterations, T,
        "$D velocity components and pressure updated; viscosity and buoyancy read",
    )
    return BenchmarkCase(
        "APT Stokes ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Stokes", ni, setup, run, validate, iterations, "iterations", parameters, model,
    )
end

"""
Linear heat diffusion with uniform unit properties, from a cold interior toward a hot bottom
and a cold top boundary; the side faces are insulated.
"""
function diffusion_state(backend, ::Val{D}, ni, ::Type{T}) where {D, T}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    li = ntuple(_ -> one(T), Val(D))
    grid = JR.Geometry(ni, li; origin = ntuple(_ -> zero(T), Val(D)))
    thermal = JR.ThermalArrays(backend, ni)
    K = device_array(backend, ones(T, ni))
    ρCp = device_array(backend, ones(T, ni))
    dt = T(1.0e-3)
    pt_thermal = JR.PTThermalCoeffs(backend, K, ρCp, dt, grid.di.center, li; CFL = 0.95 / √(D + 0.1))
    sides = D == 2 ? (left = true, right = true) : (left = true, right = true, front = true, back = true)
    thermal_bc = JR.TemperatureBoundaryConditions(;
        no_flux = merge(sides, (top = false, bot = false)),
        constant_value = merge(map(!, sides), (top = zero(T), bot = one(T))),
    )
    JR.thermal_bcs!(thermal, thermal_bc)
    return (; thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid, iterations = Ref(0))
end

function diffusion_case(backend, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    setup() = diffusion_state(backend, dims, ni, T)
    function run(state)
        (; thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid) = state
        out = JR.heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid;
            kwargs = (; igg = IGG_HANDLE[], iterMax = iterations, nout = iterations, verbose = false),
        )
        state.iterations[] = only(out.iter_count)
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("diffusion benchmark ran $(state.iterations[]) iterations instead of $iterations")
        return validate_fields("diffusion", state.thermal.T)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
    )
    model = teff_model(
        D == 2 ? 38 : 52, 1, 3, n^D * iterations, T,
        "temperature updated; old temperature, conductivity, and heat capacity read; one residual evaluation",
    )
    return BenchmarkCase(
        "PT heat diffusion ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Thermal diffusion", ni, setup, run, validate, iterations, "iterations", parameters, model,
    )
end

function benchmark_cases(
        backend = CPUBackend; size_2d = 128, size_3d = 32, iterations = 100,
        precision::Type = Float64,
    )
    return (
        stokes_case(backend, size_2d, iterations, precision, Val(2)),
        stokes_case(backend, size_3d, iterations, precision, Val(3)),
        diffusion_case(backend, size_2d, iterations, precision, Val(2)),
        diffusion_case(backend, size_3d, iterations, precision, Val(3)),
    )
end

# The solvers exchange halos through ImplicitGlobalGrid, whose global grid must match the
# case's cell counts; each case initializes it for the duration of its measurement.
const IGG_HANDLE = Ref{Any}()

function with_global_grid(f, ni)
    nx, ny, nz = length(ni) == 2 ? (ni..., 1) : ni
    IGG_HANDLE[] = JustRelax.IGG(init_global_grid(nx, ny, nz; init_MPI = !MPI.Initialized(), quiet = true)...)
    try
        return f()
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

function git_metadata()
    commit = readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`)
    status = read(`git -C $REPOSITORY_ROOT status --porcelain`, String)
    dirty = !isempty(status)
    dirty && @warn "benchmarking a dirty worktree; results will not be published" status
    subject = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%s`)
    author = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%an`)
    return (; commit, dirty, subject, author)
end

function justrelax_source()
    source = realpath(dirname(dirname(pathof(JustRelax))))
    source == realpath(REPOSITORY_ROOT) ||
        error("benchmark loaded JustRelax from $source instead of $REPOSITORY_ROOT")
    return source
end

function loaded_package_versions()
    tracked = Set(
        (
            "AMDGPU", "CUDA", "CellArrays", "GeoParams", "ImplicitGlobalGrid", "JustPIC",
            "ParallelStencil",
        ),
    )
    versions = Dict{String, String}()
    for (id, package) in Base.loaded_modules
        id.name in tracked || continue
        version = pkgversion(package)
        isnothing(version) || (versions[id.name] = string(version))
    end
    return versions
end

triad!(a::Array, b, c, s) = Threads.@threads for i in eachindex(a, b, c)
    a[i] = b[i] + s * c[i]
end
triad!(a, b, c, s) = a .= b .+ s .* c

# Eight independent chains hide FMA latency; summing them keeps them live.
function fma_chain(x, s, c, iterations)
    x1, x2, x3, x4 = x, x + 1, x + 2, x + 3
    x5, x6, x7, x8 = x + 4, x + 5, x + 6, x + 7
    for _ in 1:iterations
        x1 = muladd(x1, s, c); x2 = muladd(x2, s, c)
        x3 = muladd(x3, s, c); x4 = muladd(x4, s, c)
        x5 = muladd(x5, s, c); x6 = muladd(x6, s, c)
        x7 = muladd(x7, s, c); x8 = muladd(x8, s, c)
    end
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
end

fma_chains!(out::Array, x, s, c, iterations) = Threads.@threads for i in eachindex(out, x)
    out[i] = fma_chain(x[i], s, c, iterations)
end
fma_chains!(out, x, s, c, iterations) = out .= fma_chain.(x, s, c, iterations)

# Fastest of `repeats` calls of `f`, after `warmup` seconds of calls that let the device reach
# its sustained clock.
function best_time(f, synchronized; repeats = 5, warmup = 1.0)
    timed() = @elapsed (f(); synchronize(synchronized))
    start = time()
    while time() - start < warmup
        timed()
    end
    return minimum(_ -> timed(), 1:repeats)
end

"""
    measure_peaks(backend, T; n = 2^26, fma_items = 2^22, fma_iterations = 256)

Attainable `T` memory bandwidth (STREAM triad over `n` elements, counting 3 transfers per
element) and compute rate (8 FMA chains of `fma_iterations` per item), in GB/s and GFLOP/s.
The CPU chains run one item per loop iteration without SIMD across items, so the CPU compute
rate is a lower bound.
"""
function measure_peaks(
        backend, ::Type{T} = Float64; n = 2^26, fma_items = 2^22, fma_iterations = 256
    ) where {T}
    a, b, c = (device_array(backend, ones(T, n)) for _ in 1:3)
    bandwidth = 3 * n * sizeof(T) / best_time(() -> triad!(a, b, c, T(3)), a) / 1.0e9

    x = device_array(backend, T.(1:fma_items))
    out = similar(x)
    flops = 2 * 8 * fma_iterations * fma_items
    compute = flops / best_time(() -> fma_chains!(out, x, T(0.999), T(0.001), fma_iterations), out) / 1.0e9
    all(isfinite, Array(out)) || error("FMA probe produced non-finite values")
    return (; bandwidth, compute)
end

# Device arrays are synchronized by downloading one element, which works on every backend
# without depending on the vendor package.
synchronize(a::Array) = nothing
synchronize(a) = (Array(view(a, 1:1)); nothing)

function benchmark_metadata(backend, backend_name, device, ::Type{T}) where {T}
    backend_name == "CPU" || !isnothing(device) ||
        throw(ArgumentError("a device description is required for the $backend_name backend"))
    git = git_metadata()
    source = justrelax_source()
    cpu = first(Sys.cpu_info())
    cpu_model = "$(cpu.model) ($(Sys.CPU_NAME))"
    device = something(device, cpu_model)
    peaks = measure_peaks(backend, T)
    return Dict{String, Any}(
        "timestamp_utc" => string(now(UTC)),
        "commit" => git.commit,
        "commit_subject" => git.subject,
        "commit_author" => git.author,
        "dirty" => git.dirty,
        "julia_version" => string(VERSION),
        "justrelax_version" => string(pkgversion(JustRelax)),
        "justrelax_source" => relpath(source, REPOSITORY_ROOT),
        "backend" => backend_name,
        "float_type" => string(T),
        "device" => device,
        "cpu_model" => cpu_model,
        "hardware_fingerprint" =>
            "$(Sys.KERNEL) | $(Sys.ARCH) | $device | $cpu_model | $(Threads.nthreads()) threads",
        "threads" => Threads.nthreads(),
        "package_versions" => loaded_package_versions(),
        "peak_memory_bandwidth_gb_per_second" => peaks.bandwidth,
        "peak_compute_gflops" => peaks.compute,
        "peak_source" => "measured: $T STREAM triad and FMA-chain kernels",
    )
end

function measure(case::BenchmarkCase, samples, metadata)
    trial = with_global_grid(case.grid_cells) do
        case.validate(case.run(case.setup()))
        setup, run_case, validate = case.setup, case.run, case.validate
        @be setup run_case validate evals = 1 samples = samples seconds = Inf
    end
    timings = [sample.time for sample in trial.samples]
    median_time = median(timings)
    model = case.performance_model
    flops_per_second = model.flops / median_time

    return Dict{String, Any}(
        "name" => case.name,
        "group" => case.group,
        "unit" => "s",
        "value" => median_time,
        "time_min_seconds" => minimum(timings),
        "time_median_seconds" => median_time,
        "time_mean_seconds" => mean(timings),
        "time_max_seconds" => maximum(timings),
        "time_iqr_seconds" => quantile(timings, 0.75) - quantile(timings, 0.25),
        "samples" => length(timings),
        "allocations" => round(Int, minimum(sample -> sample.allocs, trial.samples)),
        "bytes" => round(Int, minimum(sample -> sample.bytes, trial.samples)),
        "work_units" => case.work_units,
        "work_unit" => case.work_unit,
        "throughput_per_second" => case.work_units / median_time,
        "modeled_flops" => model.flops,
        "modeled_memory_bytes" => model.memory_bytes,
        "arithmetic_intensity_flops_per_byte" => model.flops / model.memory_bytes,
        "effective_flops_per_second" => flops_per_second,
        "effective_gflops_per_second" => flops_per_second / 1.0e9,
        "effective_bandwidth_gb_per_second" => model.memory_bytes / median_time / 1.0e9,
        "performance_metric_source" => "algorithmic_model",
        "performance_model_version" => PERFORMANCE_MODEL_VERSION,
        "performance_model_description" => model.description,
        "sanity_check" => "passed",
        "parameters" => case.parameters,
        "metadata" => metadata,
    )
end

function run_benchmarks(;
        backend = CPUBackend, backend_name = "CPU", device = nothing,
        samples = 10, group = "all", precision::Type = Float64,
        cases = benchmark_cases(backend; precision),
    )
    samples > 0 || throw(ArgumentError("samples must be positive"))
    selected = group == "all" ? cases : filter(case -> case.group == group, cases)
    isempty(selected) && throw(ArgumentError("unknown or empty benchmark group: $group"))
    metadata = benchmark_metadata(backend, backend_name, device, precision)
    return map(selected) do case
        @info "Benchmarking $(case.name)" samples
        measure(case, samples, metadata)
    end
end
function write_results(path, results)
    output = abspath(path)
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, results, 2)
        write(io, '\n')
    end
    return output
end

function dashboard_run(path)
    results = JSON.parsefile(path)
    results isa Vector || error("dashboard input $path must contain a JSON array")
    isempty(results) && error("dashboard input $path contains no benchmark results")

    names = getindex.(results, "name")
    allunique(names) || error("dashboard input $path contains duplicate benchmark names")
    metadata = results[1]["metadata"]
    required = ("commit", "timestamp_utc", "backend", "hardware_fingerprint", "dirty")
    for key in required
        haskey(metadata, key) || error("dashboard input $path is missing metadata.$key")
    end
    for result in results
        result["metadata"] == metadata ||
            error("dashboard input $path mixes results from different benchmark runs")
    end

    return Dict{String, Any}(
        "source" => basename(normpath(path)),
        "commit" => metadata["commit"],
        "timestamp_utc" => metadata["timestamp_utc"],
        "backend" => metadata["backend"],
        "hardware_fingerprint" => metadata["hardware_fingerprint"],
        "dirty" => metadata["dirty"],
        "benchmarks" => results,
    )
end

function dashboard_data(inputs)
    isempty(inputs) && throw(ArgumentError("at least one dashboard input is required"))
    runs = dashboard_run.(inputs)
    identities = [(run["commit"], run["backend"], run["hardware_fingerprint"]) for run in runs]
    allunique(identities) || error(
        "dashboard inputs contain more than one run for the same commit, backend, and hardware",
    )
    sort!(runs; by = run -> run["timestamp_utc"])
    return Dict{String, Any}(
        "schema_version" => 1,
        "generated_at_utc" => string(now(UTC)),
        "repository_url" => REPOSITORY_URL,
        "runs" => runs,
    )
end

write_dashboard_data(path, inputs) = write_results(path, dashboard_data(inputs))

format_seconds(t) = t < 1.0e-3 ? "$(round(t * 1.0e6; sigdigits = 3)) μs" :
    t < 1 ? "$(round(t * 1.0e3; sigdigits = 3)) ms" : "$(round(t; sigdigits = 3)) s"

relative_spread(result) = result["time_iqr_seconds"] / result["time_median_seconds"]

format_timing(result) =
    "$(format_seconds(result["time_median_seconds"])) ±$(round(Int, 100 * relative_spread(result)))%"

"""
    print_comparison(io, baseline, candidate)

Print a markdown table of the median time of each benchmark in `candidate` relative to
`baseline`, both result vectors as written by [`write_results`](@ref). The spread is the
interquartile range as a percentage of the median. A ratio that differs from 1 by more than
the two spreads combined is marked 🔴 (slower) or 🟢 (faster). Returns the
candidate-to-baseline median ratios.
"""
function print_comparison(io::IO, baseline, candidate)
    base_meta, cand_meta = baseline[1]["metadata"], candidate[1]["metadata"]
    for key in ("backend", "float_type", "hardware_fingerprint")
        base_meta[key] == cand_meta[key] || error(
            "cannot compare runs with different $key: $(repr(base_meta[key])) vs $(repr(cand_meta[key]))",
        )
    end
    getindex.(baseline, "name") == getindex.(candidate, "name") ||
        error("baseline and candidate ran different benchmarks")

    ratios = map((b, c) -> c["time_median_seconds"] / b["time_median_seconds"], baseline, candidate)
    rows = map(baseline, candidate, ratios) do base, cand, ratio
        noise = relative_spread(base) + relative_spread(cand)
        marker = ratio > 1 + noise ? " 🔴" : ratio < 1 - noise ? " 🟢" : ""
        [
            cand["name"], format_timing(base), format_timing(cand),
            "$(base["allocations"]) → $(cand["allocations"])", "$(round(ratio; digits = 2))$marker",
        ]
    end
    header = ["Benchmark", "Baseline", "Candidate", "Allocations", "Ratio"]
    widths = [maximum(length, getindex.([[header]; collect(rows)], j)) for j in eachindex(header)]
    cells(row) = join((j == 1 ? rpad(x, w) : lpad(x, w) for (j, (x, w)) in enumerate(zip(row, widths))), " | ")
    label(meta) = "`$(first(meta["commit"], 8))`" * (meta["dirty"] ? " (dirty)" : "")

    println(io, "Baseline $(label(base_meta)) vs candidate $(label(cand_meta)) on `$(cand_meta["hardware_fingerprint"])`")
    println(io)
    println(io, "| ", cells(header), " |")
    println(io, "| :", "-"^(widths[1] - 1), " | ", join(("-"^(w - 1) * ":" for w in widths[2:end]), " | "), " |")
    foreach(row -> println(io, "| ", cells(row), " |"), rows)
    return ratios
end

const PRECISIONS = Dict("Float64" => Float64, "Float32" => Float32)

function parse_commandline(args)
    output = "benchmark_results.json"
    samples = 10
    group = "all"
    precision = Float64
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--group=")
            group = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--precision=")
            name = split(arg, '='; limit = 2)[2]
            precision = get(PRECISIONS, name) do
                throw(ArgumentError("unknown precision $(repr(name)); use --precision=Float64|Float32"))
            end
        else
            throw(
                ArgumentError(
                    "unknown argument $arg; use --output=PATH, --samples=N, --group=NAME, or --precision=TYPE"
                )
            )
        end
    end
    return (; output, samples, group, precision)
end

function main(args = ARGS; backend = CPUBackend, backend_name = "CPU", device = nothing)
    options = parse_commandline(args)
    results = run_benchmarks(;
        backend, backend_name, device, options.samples, options.group, options.precision
    )
    output = write_results(options.output, results)
    @info "Wrote benchmark results" output
    return results
end

function parse_dashboard_commandline(args)
    output = "benchmark_history.json"
    inputs = String[]
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--input=")
            push!(inputs, split(arg, '='; limit = 2)[2])
        else
            throw(ArgumentError("unknown argument $arg; use --input=PATH or --output=PATH"))
        end
    end
    isempty(inputs) && push!(inputs, "benchmark_results.json")
    return (; output, inputs)
end

function dashboard_main(args = ARGS)
    options = parse_dashboard_commandline(args)
    output = write_dashboard_data(options.output, options.inputs)
    @info "Wrote benchmark history" output
    return output
end

end
