push!(LOAD_PATH, "..")

using Test
using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/stokes3D/burstedde/Burstedde.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16
    _, _, iters = burstedde(; nx=nx, ny=ny, nz=nz, init_MPI=false, finalize_MPI=false)
    iters_expected = (
        iter=10001,
        err_evo1=[
            0.0001926888709092881,
            0.00019268875672176012,
            0.00019268875672138127,
            0.00019268875672138507,
            0.00019268875672138645,
            0.0001926887567213894,
            0.00019268875672139702,
            0.00019268875672137235,
            0.00019268875672138864,
            0.00019268875672139588,
            0.0001926887567213846,
            0.00019268875672136626,
            0.00019268875672138143,
            0.00019268875672137054,
            0.00019268875672139501,
            0.00019268875672139257,
            0.00019268875672138577,
            0.0001926887567214015,
            0.00019268875672138452,
            0.00019268875672139182,
        ],
        err_evo2=[
            500,
            1000,
            1500,
            2000,
            2500,
            3000,
            3500,
            4000,
            4500,
            5000,
            5500,
            6000,
            6500,
            7000,
            7500,
            8000,
            8500,
            9000,
            9500,
            10000,
        ],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
