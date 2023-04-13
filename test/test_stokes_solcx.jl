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
    _, _, iters, = solCx(Δη; nx=nx, ny=ny,init_MPI=true, finalize_MPI=false)
    iters_expected = (iter=3900, err_evo1=[6.26871576869803e-9])
    passed =
    iters.iter == iters_expected.iter &&
    iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
    
    return passed
end

@testset begin
    @test check_convergence_case1()
end
