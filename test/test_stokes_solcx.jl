push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes2D/solcx/SolCx.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    Δη = 1e6
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    _, _, iters, = solCx(Δη; nx=nx, ny=ny, init_MPI=init_mpi, finalize_MPI=false)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset begin
    @test check_convergence_case1()
end
