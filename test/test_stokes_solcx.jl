push!(LOAD_PATH, "..")

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

include("../miniapps/benchmarks/stokes2D/solcx/SolCx.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    Δη = 1e6
    init_MPI = JustRelax.MPI.Initialized() ? false : true
    _, _, iters, = solCx(Δη; nx=nx, ny=ny, init_MPI=init_MPI, finalize_MPI=false)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "solcx" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
