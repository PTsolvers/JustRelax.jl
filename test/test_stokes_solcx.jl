push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using ParallelStencil
    @init_parallel_stencil(AMDGPU, Float64, 2)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using ParallelStencil
    @init_parallel_stencil(CUDA, Float64, 2)
else
    using ParallelStencil
    @init_parallel_stencil(Threads, Float64, 2)
end

include("../miniapps/benchmarks/stokes2D/solcx/SolCx.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    Δη = 1e6
    init_MPI = JustRelax.MPI.Initialized() ? false : true
    _, _, iters, = solCx(Δη; nx=nx, ny=ny, init_MPI=init_MPI, finalize_MPI=false)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "solcx" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
