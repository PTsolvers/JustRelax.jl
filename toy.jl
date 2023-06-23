struct Foo2{T}
    xx::T
    yy::T
    xy::T
end

struct Foo3{T}
    xx::T
    yy::T
    zz::T
    yz::T
    xz::T
    xy::T
end

A = @rand(2, 2)
B = @rand(2, 2, 2)

foo2 = Foo2(A, A, A)
foo3 = Foo3(B, B, B, B, B, B)

function f1(A::SymmetricTensor{<:AbstractArray{T, N}}) where {T, N}
    syms = (:xx ,:yy, :zz)
    return ntuple(i -> getfield(A, syms[i]), N)
end

@generated function f2(A::SymmetricTensor{<:AbstractArray{T, N}}) where {T, N}
    syms = (:xx ,:yy, :zz)
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> f_i = getfield(A, $syms[i])
        Base.@ncall $N tuple f
    end
end

@inline function f3(
    A::SymmetricTensor{<:AbstractArray{T,2}}
) where {T}
    return A.xx, A.yy
end

@inline function f4(
    A::SymmetricTensor{<:AbstractArray{T,3}}
) where {T}
    return A.xx, A.yy, A.zz
end


@btime f1($A);
@btime f2($A);
@btime f4($A);

@code_lowered f1(A)
@code_lowered f2(A)
@code_lowered f4(A)
