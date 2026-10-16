using Pkg

const BENCHMARK_PROJECT = realpath(@__DIR__)
const ACTIVE_PROJECT = realpath(dirname(Base.active_project()))

ACTIVE_PROJECT == BENCHMARK_PROJECT ||
    error("activate the benchmark environment with --project=benchmarking")

# A relative path keeps the tracked `[sources]` entry of Project.toml unchanged.
Pkg.develop(; path = relpath(dirname(@__DIR__)))
Pkg.instantiate()
