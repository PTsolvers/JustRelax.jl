# Runs the VanKeken DYREL resolution study sequentially (32 -> 1024).
# The highest resolution also gets 3 density-field snapshot checkpoints.
#
# Usage: julia -t 8 --project=miniapps miniapps/DYREL2D/VanKeken/run_resolution_study.jl
# (run from the repo root; each resolution runs as its own subprocess, since
# VanKeken_DYREL.jl does its own @init_parallel_stencil and can't be `include`d twice.)

cd(joinpath(@__DIR__, "..", "..", ".."))  # repo root (miniapps/Project.toml lives here)

nthreads = Threads.nthreads()
julia_exe = Base.julia_cmd()
resolutions = (32, 64, 128, 256, 512, 1024)
nmax = maximum(resolutions)

for n in resolutions
    println("=== VanKeken DYREL: $(n)x$(n) ===")
    args = n == nmax ? ("$n", "snapshots") : ("$n",)
    run(`$julia_exe --project -t $nthreads miniapps/DYREL2D/VanKeken/VanKeken_DYREL.jl $args`)
end
