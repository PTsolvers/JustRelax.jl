push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes/solcx/SolCx.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    Δη = 1e6
    _, _, iters, = solCx(Δη; nx=nx, ny=ny);
    iters_expected = (iter=150001, err_evo1=[3.427298333287185e-8])
    # println(iters)
    # println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
