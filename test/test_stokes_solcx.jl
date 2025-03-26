push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

include("../miniapps/benchmarks/stokes2D/solcx/SolCx.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    Δη = 1.0e6
    init_MPI = JustRelax.MPI.Initialized() ? false : true
    _, _, iters, = solCx(Δη; nx = nx, ny = ny, init_MPI = init_MPI, finalize_MPI = false)

    tol = 1.0e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "solcx" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
