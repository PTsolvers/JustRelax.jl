push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes/solkz/SolKz.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    _, _, iters, _ = solKz(; nx=nx, ny=ny)
    iters_expected = (iter=3000, err_evo1=[4.813927034774679e-13])
    return iters.iter == iters_expected.iter && iters.err_evo1[end] < 1e-6
end

@testset begin
    @test check_convergence_case1()
end
