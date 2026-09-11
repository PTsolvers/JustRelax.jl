# DYREL 3D benchmarks

The examples assume the REPL's working directory is the root of the JustRelax repository,
so the relative `include(...)` path resolves. The benchmark runner activates the
`miniapps` environment itself.

## Select a benchmark

Edit `benchmark` near the top of `RunStokesDYRELBench3D.jl`. Available values are:

| Value | Benchmark | MPI-aware validation |
| --- | --- | --- |
| `:taylorGreen` | Taylor–Green analytical solution | Yes |
| `:Burstedde` | Burstedde mantle-convection benchmark | No |
| `:solvi` | SolVi viscous-inclusion benchmark | No |

The runner currently uses the CPU/Threads backend. Set `nx`, `ny`, and `nz` in the same
file to change the local resolution.

## Run a benchmark in the REPL

Start a Julia REPL from the repository root. Then run:

```julia
include("miniapps/DYREL3D/benchmarks/RunStokesDYRELBench3D.jl")
```

A normal REPL run uses one rank. Restart the REPL before selecting and running another
benchmark, because ParallelStencil is initialized when the runner is included.

## Run the Taylor–Green MPI test

Only Taylor–Green currently has MPI-aware error validation and an automated multi-rank
test. Burstedde and SolVi should be run on one rank.

```bash
JULIA_JUSTRELAX_BACKEND=CPU ~/.julia/bin/mpiexecjl -n 2 \
  julia --project=. --startup-file=no test/test_dyrel_taylor_green_MPI.jl
```

This test checks that more than one rank is active, that DYREL converges, and that the
global pressure and velocity errors remain below their tolerances. The repository test
runner also discovers files containing `MPI` in their name and launches them with two
ranks automatically.
