# Collect the γ sweeps written by `drive_tune.sh` and report, per solver and resolution, the
# γfact that costs the fewest dynamic-relaxation iterations.
#
#   julia --project=<thisdir> <thisdir>/report_tune.jl tune2_n64 tune2_n128 tune2_n256
#
# Only candidates that ran every step and converged on all of them are ranked: an unconverged
# step stops at `total_iterMax` and a diverged one stops earlier still, so both have a censored
# cost that would otherwise look cheap next to a candidate that actually finished.

using Pkg
Pkg.activate(@__DIR__)
using DelimitedFiles, Printf

const DIRS = isempty(ARGS) ? ["tune2_n64", "tune2_n128", "tune2_n256"] : ARGS

struct Row
    solver::String
    N::Int
    γ::Float64
    steps::Int
    total::Int
    mean::Float64
    maxi::Int
    unconv::Int
    secs::Float64
    status::String
end

function load(dir)
    path = joinpath(@__DIR__, dir, "gamma_sweep.csv")
    isfile(path) || return Row[]
    data, _ = readdlm(path, ',', header = true)
    return [
        Row(
                string(data[i, 1]), Int(data[i, 2]), Float64(data[i, 3]), Int(data[i, 4]),
                Int(data[i, 5]), Float64(data[i, 6]), Int(data[i, 8]), Int(data[i, 9]),
                Float64(data[i, 10]), string(data[i, 11]),
            ) for i in axes(data, 1)
    ]
end

rows = reduce(vcat, load.(DIRS))
isempty(rows) && error("no sweep CSVs found in $(DIRS)")

# `steps` is the run length the sweep was configured with; take it from the candidates that ran.
const NSTEPS = maximum(r -> r.steps, rows)
ranked(r) = r.status == "ok" && r.unconv == 0 && r.steps == NSTEPS

println("γfact tuning — Rayleigh-Taylor, $NSTEPS steps, fixed dt, ϵ = 1e-6, CUDA")
println("metric: total dynamic-relaxation iterations (lower is better)\n")

for solver in ("standard", "variational"), N in sort(unique(r.N for r in rows))
    sub = sort(filter(r -> r.solver == solver && r.N == N, rows), by = r -> r.γ)
    isempty(sub) && continue

    @printf("%s  %d×%d\n", uppercase(solver), N, N)
    @printf("  %8s %12s %12s %12s %10s\n", "γfact", "total iter", "mean/step", "max/step", "status")
    ok = filter(ranked, sub)
    best = isempty(ok) ? nothing : argmin(r -> r.total, ok)
    for r in sub
        mark = r === best ? "  <== best" : ""
        if ranked(r)
            @printf("  %8.4g %12d %12.0f %12d %10s%s\n", r.γ, r.total, r.mean, r.maxi, "ok", mark)
        else
            @printf("  %8.4g %12s %12s %12s %10s\n", r.γ, "-", "-", "-", r.status)
        end
    end
    if best === nothing
        println("  no γ completed all $NSTEPS steps\n")
    else
        worst = argmax(r -> r.total, ok)
        @printf(
            "  best γfact = %.4g  (%d iters, %.0f/step); %.2f× cheaper than γ=%.4g\n\n",
            best.γ, best.total, best.mean, worst.total / best.total, worst.γ
        )
    end
end

println("SUMMARY — best γfact")
@printf("  %-14s %10s %14s %14s\n", "solver", "N", "best γfact", "iters/step")
for solver in ("standard", "variational"), N in sort(unique(r.N for r in rows))
    ok = filter(r -> ranked(r) && r.solver == solver && r.N == N, rows)
    isempty(ok) && continue
    b = argmin(r -> r.total, ok)
    @printf("  %-14s %10d %14.4g %14.0f\n", solver, N, b.γ, b.mean)
end
