push!(LOAD_PATH, "..")

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
        iter=1000,
        err_evo1=[5.6947513423598755e-6, 2.0642133108711034e-9],
        err_evo2=[500, 1000],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
