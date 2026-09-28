# Benchmarks a base revision and the working tree back to back on this machine and prints
# the candidate-to-baseline ratios. The base revision runs the working tree's benchmark
# harness, so both sides execute identical cases. Arguments other than `--rev` are passed
# to `run_benchmarks.jl` for both runs.
using JSON: JSON
using JustRelaxBenchmarks

const REV_FLAGS = filter(startswith("--rev="), ARGS)
const REV = isempty(REV_FLAGS) ? "main" : split(only(REV_FLAGS), '='; limit = 2)[2]
any(startswith("--output="), ARGS) &&
    error("compare.jl does not write result files; drop --output")
const RUN_ARGS = filter(!startswith("--rev="), ARGS)

const ROOT = JustRelaxBenchmarks.REPOSITORY_ROOT
const JULIA = `$(Base.julia_cmd()) --threads=$(Threads.nthreads())`

mktempdir() do dir
    worktree = joinpath(dir, "baseline")
    run(`git -C $ROOT worktree add --detach --quiet $worktree $REV`)
    try
        harness = joinpath(worktree, "benchmarking")
        rm(harness; force = true, recursive = true)
        cp(joinpath(ROOT, "benchmarking"), harness)
        run(`$JULIA --project=$harness $harness/setup.jl`)

        function run_suite(project, name)
            output = joinpath(dir, "$name.json")
            run(`$JULIA --project=$project $project/run_benchmarks.jl $RUN_ARGS --output=$output`)
            return JSON.parsefile(output)
        end
        baseline = run_suite(harness, "baseline")
        candidate = run_suite(joinpath(ROOT, "benchmarking"), "candidate")
        print_comparison(stdout, baseline, candidate)
    finally
        run(`git -C $ROOT worktree remove --force $worktree`)
    end
end
