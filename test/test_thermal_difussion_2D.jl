push!(LOAD_PATH, "..")
using Pkg; Pkg.activate("C:\\Users\\albert\\Desktop\\JustRelax.jl")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/thermal_diffusion/diffusion/diffusion2D.jl")

function check_convergence_case1()
    nx = ny = 64
    L = 100e3
    _, _, iters = diffusion_2D(; nx=nx, ny=ny, lx=L, ly=L, ρ=3.3e3, Cp=1.2e3, K=3.0)
    iters_expected = (
        iter=4,
        err_evo1=[
            6.235828801688147e-7,
            1.6845474834774204e-7,
            4.293863309105398e-8,
            9.474024340600134e-9,
        ],
        err_evo2=[1, 2, 3, 4],
    )
    println(iters)
    println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
