using JSON: JSON
using JustRelax
using JustRelaxBenchmarks
using Test

@testset "benchmark harness" begin
    cases = benchmark_cases(; size_2d = 16, size_3d = 8, iterations = 10)
    results = run_benchmarks(; samples = 1, cases)

    @test length(results) == 4
    @test count(result -> result["parameters"]["dimension"] == 3, results) == 2
    @test all(result -> result["value"] > 0, results)
    @test all(result -> result["sanity_check"] == "passed", results)
    @test all(result -> result["samples"] == 1, results)
    @test all(result -> result["modeled_flops"] > 0, results)
    @test all(result -> result["modeled_memory_bytes"] > 0, results)
    @test all(result -> result["arithmetic_intensity_flops_per_byte"] > 0, results)
    @test all(result -> result["effective_gflops_per_second"] > 0, results)
    @test all(
        result -> isapprox(
            result["effective_flops_per_second"],
            result["modeled_flops"] / result["time_median_seconds"],
        ),
        results,
    )
    @test all(result -> result["effective_bandwidth_gb_per_second"] > 0, results)
    @test all(result -> result["performance_metric_source"] == "algorithmic_model", results)
    @test all(
        result -> isapprox(
            result["effective_bandwidth_gb_per_second"],
            result["modeled_memory_bytes"] / result["time_median_seconds"] / 1.0e9,
        ),
        results,
    )
    @test all(result -> result["metadata"]["justrelax_source"] == ".", results)
    @test all(result -> haskey(result["metadata"], "commit_subject"), results)
    @test all(result -> result["metadata"]["backend"] == "CPU", results)
    @test all(result -> result["metadata"]["float_type"] == "Float64", results)
    @test all(result -> endswith(result["name"], "Float64)"), results)
    @test results[1]["metadata"]["peak_memory_bandwidth_gb_per_second"] > 0
    @test results[1]["metadata"]["peak_compute_gflops"] > 0
    @test_throws "device description is required" run_benchmarks(;
        backend_name = "CUDA", samples = 1, cases,
    )
    @test_throws "unknown precision" JustRelaxBenchmarks.parse_commandline(["--precision=Float16"])

    comparison = sprint(io -> @test all(==(1), print_comparison(io, results, results)))
    @test occursin(results[1]["name"], comparison)
    @test !occursin("🔴", comparison)
    slower = deepcopy(results)
    slower[1]["time_median_seconds"] *= 2
    @test occursin("🔴", sprint(print_comparison, results, slower))
    elsewhere = deepcopy(results)
    elsewhere[1]["metadata"]["hardware_fingerprint"] = "another machine"
    @test_throws "different hardware_fingerprint" print_comparison(devnull, results, elsewhere)

    mktempdir() do dir
        path = write_results(joinpath(dir, "results.json"), results)
        decoded = JSON.parsefile(path)
        @test getindex.(decoded, "name") == collect(getindex.(results, "name"))

        history = write_dashboard_data(joinpath(dir, "history.json"), [path])
        payload = JSON.parsefile(history)
        @test payload["schema_version"] == 1
        @test payload["repository_url"] == "https://github.com/PTsolvers/JustRelax.jl"
        @test length(only(payload["runs"])["benchmarks"]) == 4
        @test_throws "more than one run for the same commit" write_dashboard_data(
            joinpath(dir, "duplicate-history.json"),
            [path, path],
        )
    end
end
