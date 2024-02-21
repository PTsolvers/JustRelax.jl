push!(LOAD_PATH, "..")

using Test, Suppressor
using JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/stokes3D/taylor_green/TaylorGreen.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16

    # run model
    _, _, iters = taylorGreen(;
        nx=nx,
        ny=ny,
        nz=ny,
        init_MPI=JustRelax.MPI.Initialized() ? false : true,
        finalize_MPI=false,
    )

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "TaylorGreen" begin
    @test check_convergence_case1()
end
