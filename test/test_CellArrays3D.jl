
using Test, StaticArrays, AllocCheck
using Suppressor

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

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

@testset "CellArrays 3D" begin
    @suppress begin
        ni = 5, 5, 5
        A  = @fill(false, ni..., celldims=(2,), eltype=Bool)

        @test @cell(A[1, 1, 1, 1]) === false
        @test (@allocated @cell A[1, 1, 1, 1]) == 0
    ;
        @cell A[1, 1, 1, 1] = true
        @test @cell(A[1, 1, 1, 1]) === true
        @test (@allocated @cell A[1, 1, 1, 1] = true) == 0

        @test A[1, 1, 1] == SA[true, false]
        # allocs = check_allocs(getindex, (typeof(A), Int64, Int64, Int64))
        # @test isempty(allocs)
    end
end
