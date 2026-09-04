# Runs the Blankenbach DYREL resolution study sequentially (32 -> 256).
#
# Usage: julia -t 8 --project=miniapps miniapps/DYREL2D/Blankenbach2D/run_resolution_study.jl
# (run from the repo root; each resolution runs as its own subprocess, since
# Benchmark2D_DYREL.jl does its own @init_parallel_stencil and can't be `include`d twice.)

cd(joinpath(@__DIR__, "..", "..", ".."))  # repo root (miniapps/Project.toml lives here)

nthreads = Threads.nthreads()
julia_exe = Base.julia_cmd()
resolutions = (32, 64, 128, 256)
nmax = maximum(resolutions)

for n in resolutions
    println("=== Blankenbach DYREL: $(n)x$(n) ===")
    args = n == nmax ? ("$n", "snapshots") : ("$n",)
    run(`$julia_exe --project -t $nthreads miniapps/DYREL2D/Blankenbach2D/Benchmark2D_DYREL.jl $args`)
end
