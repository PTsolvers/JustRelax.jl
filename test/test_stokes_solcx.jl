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
    geometry, stokes, iters, ρ = solCx(Δη; nx=nx, ny=ny)
    iters_expected = (
        iter=4000,
        err_evo1=[
            2.6882547170954827,
            0.2473522008680851,
            0.01130098855514106,
            0.0004922739546726743,
            2.139794848519281e-5,
            9.300304245694199e-7,
            4.0422237837643596e-8,
            1.7568857404566392e-9,
        ],
        err_evo2=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0],
    )
    println(iters)
    println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
