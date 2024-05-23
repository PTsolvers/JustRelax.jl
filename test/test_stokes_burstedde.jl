push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax3D

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using ParallelStencil
    @init_parallel_stencil(AMDGPU, Float64, 3)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using ParallelStencil
    @init_parallel_stencil(CUDA, Float64, 3)
else
    using ParallelStencil
    @init_parallel_stencil(Threads, Float64, 3)
end

include("../miniapps/benchmarks/stokes3D/burstedde/Burstedde.jl")

function check_convergence_case1()
    nx = 16
    ny = 16
    nz = 16
    _, _, iters = burstedde(; nx=nx, ny=ny, nz=nz, init_MPI=true, finalize_MPI=true)

    tol = 1e-8
    passed = iters.err_evo1[end] < tol

    return passed
end

@testset "Burstedde" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
