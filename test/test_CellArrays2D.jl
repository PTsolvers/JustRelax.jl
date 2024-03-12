using Test, Suppressor, StaticArrays, AllocCheck, JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

model = PS_Setup(:Threads, Float64, 2)
environment!(model)

@testset "CellArrays 2D" begin
    # @suppress begin
        ni = 5, 5
        A  = JustRelax.@fill(false, ni..., celldims=(2,), eltype=Bool)

        @test cellaxes(A) === Base.OneTo(2)
        @test cellnum(A) == 2
        @test new_empty_cell(A) === SA[false, false]

        @test @cell(A[1, 1, 1]) === false
        @test (@allocated @cell A[1, 1, 1]) === 0

        @cell A[1, 1, 1] = true
        @test @cell(A[1, 1, 1]) === true
        @test (@allocated @cell A[1, 1, 1] = true) === 0
    # end
    # @test A[1, 1] === SA[true, false]
    # allocs = check_allocs(getindex, (typeof(A), Int64, Int64))
    # @test isempty(allocs)
end
