push!(LOAD_PATH, "..")

using Test
using JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/stokes3D/burstedde/Burstedde.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16
    _, _, iters = burstedde(; nx=nx, ny=ny, nz=nz, init_MPI=true, finalize_MPI=true)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "Burstedde" begin
    @test check_convergence_case1()
end
