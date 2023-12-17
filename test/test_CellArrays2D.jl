using Test, StaticArrays, AllocCheck, JustRelax
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

@testset "CellArrays 2D" begin
    ni = 5, 5
    A  = JustRelax.@fill(false, ni..., celldims=(2,), eltype=Bool) 

    @test @cell(A[1, 1, 1]) === false
    @test (@allocated @cell A[1, 1, 1]) == 0

    @cell A[1, 1, 1] = true
    @test @cell(A[1, 1, 1]) === true
    @test (@allocated @cell A[1, 1, 1] = true) == 0

    @test A[1, 1] == SA[true, false]
    allocs = check_allocs(getA, (typeof(A), Int64, Int64))
    @test isempty(allocs)
end