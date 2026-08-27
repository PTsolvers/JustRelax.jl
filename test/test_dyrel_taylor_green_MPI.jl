push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustPIC
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    CPUBackend
end

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

include(joinpath(
    @__DIR__, "..", "miniapps", "DYREL3D", "benchmarks", "taylor_green",
    "TaylorGreen_DYREL.jl",
))

function check_taylor_green_mpi()
    geometry, stokes, iters = taylorGreen(;
        nx = 8,
        ny = 8,
        nz = 8,
        init_MPI = true,
        finalize_MPI = false,
        verbose = false,
    )
    errors = error_norms(stokes, geometry)

    return iters, errors
end

@testset "DYREL TaylorGreen MPI" begin
    iters, errors = check_taylor_green_mpi()
    @test JustRelax.MPI.Comm_size(JustRelax.MPI.COMM_WORLD) > 1
    @test !isempty(iters.err_evo_tot)
    @test all(isfinite, iters.err_evo_tot)
    @test last(iters.err_evo_tot) < first(iters.err_evo_tot)
    @test all(isfinite, errors)

    L2_p, L2_vx, L2_vy, L2_vz = errors
    @test max(L2_vx, L2_vy, L2_vz) < 5.0e-2
    @test L2_p < 1.0
end
