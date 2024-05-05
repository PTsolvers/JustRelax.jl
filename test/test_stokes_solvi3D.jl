push!(LOAD_PATH, "..")

using Test, Suppressor
using JustRelax, JustRelax.JustRelax3D
const backend = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

include("../miniapps/benchmarks/stokes3D/solvi/SolVi3D.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16

    # model specific parameters
    Δη = 1e-3 # viscosity ratio between matrix and inclusion
    rc = 1e0 # radius of the inclusion
    εbg = 1e0 # background strain rate
    lx, ly, lz = 1e1, 1e1, 1e1 # domain siye in x and y directions

    # run model
    _, _, iters = solVi3D(;
        Δη=Δη,
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        lz=lz,
        rc=rc,
        εbg=εbg,
        init_MPI=JustRelax.MPI.Initialized() ? false : true,
        finalize_MPI=false,
    )

    tol = 1e-8
    passed = iters.norm_Rx[end] < tol

    return passed
end

@testset "solvi3D" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
