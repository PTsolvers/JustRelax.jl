push!(LOAD_PATH, "..")
using Pkg: Pkg;

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 1)
environment!(model)

include("../miniapps/benchmarks/thermal_diffusion/diffusion/diffusion1D.jl")

function check_convergence_case1()
    nx = 64
    _, _, iters = diffusion_1D(; nx=nx, lx=100e3, ρ=3.3e3, Cp=1.2e3, K=3.0)
    iters_expected = (
        iter=6,
        err_evo1=[
            1.1223667373852162e-6,
            3.98750310210544e-7,
            1.3931091471554355e-7,
            4.7260089625293e-8,
            1.5183656218468635e-8,
            4.371047903932851e-9,
        ],
        err_evo2=[1, 2, 3, 4, 5, 6],
    )
    println(iters)
    println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
