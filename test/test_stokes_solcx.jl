push!(LOAD_PATH, "..")
using Pkg; Pkg.activate("C:\\Users\\albert\\Desktop\\JustRelax.jl")

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
        iter=3000,
        err_evo1=[
            0.09360331957426424,
            0.003306891096559515,
            0.00011424277131348146,
            3.946913923776043e-6,
            1.363780256933486e-7,
            4.71254016862045e-9,
        ],
        err_evo2=[500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
    )
    println(iters)
    println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
