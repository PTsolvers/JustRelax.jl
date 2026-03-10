using Test
using JustRelax, JustRelax.JustRelax2D
import JustRelax: Mask, apply_mask, apply_mask!
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

@testset "Mask 2D" begin
    ni = 10, 10

    # Test basics
    m = Mask(ni...)

    @test size(m) == ni
    @test length(m) == prod(ni)
    @test axes(m) == (Base.OneTo(ni[1]), Base.OneTo(ni[2]))
    @test eachindex(m) == Base.OneTo(prod(ni))
    @test similar(m) isa Mask{Matrix{Float64}}
    @test !all(m)

    m = JR2.Mask(ni..., 4:7, 4:7)
    @test all(isone, m[4:7, 4:7])

    # test masking
    A = rand(ni...)
    B = zeros(ni...)
    B[4:7, 4:7] .= 5
    m = JR2.Mask(ni..., 4:7, 4:7)

    C = apply_mask(A, B, m)
    @test all(C[4:7, 4:7] .== 5)

    apply_mask!(A, B, m)
    @test all(A[4:7, 4:7] .== 5)

    A = rand(ni...)
    @test apply_mask(A, B, m, 1, 1) == A[1, 1]
    @test apply_mask(A, B, m, 5, 5) == 5

    apply_mask!(A, B, m, 1, 1)
    apply_mask!(A, B, m, 5, 5)

    @test A[1, 1] != 5
    @test A[5, 5] == 5

    @test isone(inv(m, 1, 1))
    @test iszero(inv(m, 5, 5))
end

@testset "Mask 3D" begin
    ni = 10, 10, 10

    m = Mask(ni...)
    @test size(m) == ni
    @test length(m) == prod(ni)
    @test axes(m) == (Base.OneTo(ni[1]), Base.OneTo(ni[2]), Base.OneTo(ni[3]))
    @test eachindex(m) == Base.OneTo(prod(ni))
    @test similar(m) isa Mask{Array{Float64, 3}}
    @test !all(m)

    m = JR3.Mask(ni..., 4:7, 4:7, 4:7)
    @test all(isone, m[4:7, 4:7, 4:7])

    A = rand(ni...)
    B = zeros(ni...)
    B[4:7, 4:7, 4:7] .= 5
    m = JR3.Mask(ni..., 4:7, 4:7, 4:7)

    C = apply_mask(A, B, m)
    @test all(C[4:7, 4:7, 4:7] .== 5)

    apply_mask!(A, B, m)
    @test all(A[4:7, 4:7, 4:7] .== 5)

    A = rand(ni...)
    @test apply_mask(A, B, m, 1, 1, 1) == A[1, 1, 1]
    @test apply_mask(A, B, m, 5, 5, 5) == 5

    apply_mask!(A, B, m, 1, 1, 1)
    apply_mask!(A, B, m, 5, 5, 5)

    @test A[1, 1, 1] != 5
    @test A[5, 5, 5] == 5
end
