push!(LOAD_PATH, "..")
using Pkg; Pkg.activate("C:\\Users\\albert\\Desktop\\JustRelax.jl")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)s
environment!(model)

include("../miniapps/benchmarks/stokes/solkz/SolKz.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    geometry, stokes, iters, ρ = solKz(; nx=nx, ny=ny)
    iters_expected = (
        iter=3500,
        err_evo1=[
            0.004234955041985375,
            4.731816473180695e-5,
            5.595230306874109e-6,
            7.675634065893899e-7,
            1.0528689841488961e-7,
            1.4441981440483245e-8,
            1.980974468802502e-9,
        ],
        err_evo2=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
