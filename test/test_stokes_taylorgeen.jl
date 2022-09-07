push!(LOAD_PATH, "..")
using Pkg: Pkg;

using Test
using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/stokes3D/taylor_green/TaylorGreen.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16
    _, _, iters = taylorGreen(; nx=nx, ny=ny, nz=nz, init_MPI=true, finalize_MPI=true)
    iters_expected = (
        iter=1500,
        err_evo1=[
            3.863567605285833e-5
            1.4614649338896292e-8
            6.086219718431415e-12
        ],
        err_evo2=[500, 1000, 1500],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
