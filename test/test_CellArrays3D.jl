
using Test, StaticArrays, AllocCheck
using Suppressor
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

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
