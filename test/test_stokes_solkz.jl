push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes/solkz/SolKz.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    _, _, iters, _ = solKz(; nx=nx, ny=ny,init_MPI=true, finalize_MPI=false)
    iters_expected = (iter=3000, err_evo1=[4.813927034774679e-13])
    passed =
    iters.iter == iters_expected.iter &&
    iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
    
    return passed
end

@testset begin
    @test check_convergence_case1()
end
