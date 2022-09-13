push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes/solkz/SolKz.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    geometry, stokes, iters, ρ = solKz(; nx=nx, ny=ny)
    iters_expected = (
        iter=3500,
        err_evo1=[
            0.0042919263496513805,
            4.650387644954902e-5,
            5.435263102449807e-6,
            7.456389613594398e-7,
            1.0228337332779032e-7,
            1.4030512629915123e-8,
            1.924605146654628e-9,
        ],
        err_evo2=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
