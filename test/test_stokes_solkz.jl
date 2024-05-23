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
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    JustRelax.CUDABackend
else
    JustRelax.CPUbackend
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


include("../miniapps/benchmarks/stokes2D/solkz/SolKz.jl")

function check_convergence_case1()
    nx = 64
    ny = 64
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    _, _, iters, _ = solKz(; nx=nx, ny=ny, init_MPI=init_mpi, finalize_MPI=false)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "solkz" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
