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
    _, _, iters = burstedde(; nx=nx, ny=ny, nz=nz, init_MPI=false, finalize_MPI=false);
    iters_expected = (
        iter=20001,
        err_evo1=[0.0027847862195855555],
    )
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
