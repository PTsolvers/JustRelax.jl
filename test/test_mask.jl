using Test

ni = 10, 10

# Test basics 
m  = Mask(ni...)

@test size(m)      == ni
@test length(m)    == prod(ni)
@test axes(m)      == Base.OneTo(ni), Base.OneTo(ni)
@test eachindex(m) == Base.OneTo(prod(ni))
@test similar(m)   isa Mask{Matrix{Float64}}
@test !all(m) 

m  = Mask(ni..., 4:7, 4:7)
@test all(isone, m[4:7, 4:7])

# test masking
A  = rand(ni...)
B  = zeros(ni...)
B[4:7, 4:7] .= 5
m  = Mask(ni..., 4:7, 4:7)

C = apply_mask(A, B, m)
@test all(C[4:7, 4:7] .== 5)

apply_mask!(A, B, m)
@test all(A[4:7, 4:7] .== 5)

A  = rand(ni...)
@test apply_mask(A, B, m, 1, 1) == A[1, 1]
@test apply_mask(A, B, m, 5, 5) == 5

apply_mask!(A, B, m, 1, 1)
apply_mask!(A, B, m, 5, 5)

@test A[1, 1] != 5
@test A[5, 5] == 5

@test isone(inv(m, 1, 1))
@test iszero(inv(m, 5, 5))