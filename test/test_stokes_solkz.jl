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
            0.0044223308158044565,
            4.944247108047785e-5,
            5.9105921832100726e-6,
            8.263947237140499e-7,
            1.1553438985652414e-7,
            1.6152029093324383e-8,
            2.2580967107291683e-9,
        ],
        err_evo2=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
