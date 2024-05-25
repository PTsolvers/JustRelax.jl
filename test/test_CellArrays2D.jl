@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
    AMDGPU.allowscalar(true)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
    CUDA.allowscalar(true)
end

using Test, StaticArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@testset "CellArrays 2D" begin
    ni = 5, 5
    A  = @fill(false, ni..., celldims=(2,), eltype=Bool)

    @test cellaxes(A)       === Base.OneTo(2)
    @test cellnum(A)        == 2
    @test new_empty_cell(A) === SA[false, false]

    @test @cell(A[1, 1, 1]) === false
    # @test (@allocated @cell A[1, 1, 1]) === 0

    @cell A[1, 1, 1] = true
    @test @cell(A[1, 1, 1])                    === true
    # @test (@allocated @cell A[1, 1, 1] = true) === 0
    @test A[1, 1]                              === SA[true, false]
end
