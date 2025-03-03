# masked versions
for fn in (:center, :next, :left, :right, :back, :front)
    @eval begin
        Base.@propagate_inbounds @inline ($fn)(
            A::T, ϕ::T, inds::Vararg{Integer, N}
        ) where {T <: AbstractArray, N} = ($fn)(A, inds...) * ($fn)(ϕ, inds...)
    end
end

# finite differences
Base.@propagate_inbounds @inline _d_xa(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} =
    (-center(A, ϕ, I...) + right(A, ϕ, I...)) * _dx

Base.@propagate_inbounds @inline _d_ya(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} =
    (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dy

Base.@propagate_inbounds @inline _d_za(A::T, ϕ::T, _dz, I::Vararg{Integer, N}) where {N, T} =
    (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dz

Base.@propagate_inbounds @inline _d_xi(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} =
    (-front(A, ϕ, I...) + next(A, ϕ, I...)) * _dx

Base.@propagate_inbounds @inline _d_yi(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} =
    (-right(A, ϕ, I...) + next(A, ϕ, I...)) * _dy

Base.@propagate_inbounds @inline _d_zi(A::T, ϕ::T, _dz, I::Vararg{Integer, N}) where {N, T} =
    (-top(A, ϕ, I...) + next(A, ϕ, I...)) * _dz

# averages 2D
Base.@propagate_inbounds @inline _av(A::T, ϕ::T, i, j) where {T <: T2} =
    0.25 * mysum(A, ϕ, (i + 1):(i + 2), (j + 1):(j + 2))

Base.@propagate_inbounds @inline _av_a(A::T, ϕ::T, i, j) where {T <: T2} =
    0.25 * mysum(A, ϕ, (i):(i + 1), (j):(j + 1))

Base.@propagate_inbounds @inline _av_xa(A::T, ϕ::T, I::Vararg{Integer, 2}) where {T <: T2} =
    (center(A, ϕ, I...) + right(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_ya(A::T, ϕ::T, I::Vararg{Integer, 2}) where {T <: T2} =
    (center(A, ϕ, I...) + front(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_xi(A::T, ϕ::T, I::Vararg{Integer, 2}) where {T <: T2} =
    (front(A, ϕ, I...), next(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_yi(A::T, ϕ::T, I::Vararg{Integer, 2}) where {T <: T2} =
    (right(A, ϕ, I...), next(A, ϕ, I...)) * 0.5

# averages 3D
Base.@propagate_inbounds @inline _av(A::T, ϕ::T, i, j, k) where {T <: T3} =
    0.125 * mymaskedsum(A, ϕ, (i + 1):(i + 2), (j + 1):(j + 2), (k + 1):(k + 2))

Base.@propagate_inbounds @inline _av_a(A::T, ϕ::T, i, j, k) where {T <: T3} =
    0.125 * mymaskedsum(A, ϕ, (i):(i + 1), (j):(j + 1), (k):(k + 1))

Base.@propagate_inbounds @inline _av_xa(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (center(A, ϕ, I...) + right(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_ya(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (center(A, ϕ, I...) + front(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_za(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (center(A, ϕ, I...) + top(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_xi(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (front(A, ϕ, I...), next(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_yi(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (right(A, ϕ, I...), next(A, ϕ, I...)) * 0.5

Base.@propagate_inbounds @inline _av_zi(A::T, ϕ::T, I::Vararg{Integer, 3}) where {T <: T3} =
    (top(A, ϕ, I...) + next(A, ϕ, I...)) * 0.5

## Because mymaskedsum(::generator) does not work inside CUDA kernels...
@inline mymaskedsum(A::AbstractArray, ϕ::AbstractArray, ranges::Vararg{T, N}) where {T, N} =
    mymaskedsum(identity, A, ϕ, ranges...)

@inline function mymaskedsum(
        f::F, A::AbstractArray, ϕ::AbstractArray, ranges_i
    ) where {F <: Function}
    s = 0.0
    for i in ranges_i
        s += f(A[i]) * ϕ[i]
    end
    return s
end

@inline function mymaskedsum(
        f::F, A::AbstractArray, ϕ::AbstractArray, ranges_i, ranges_j
    ) where {F <: Function}
    s = 0.0
    for i in ranges_i, j in ranges_j
        s += f(A[i, j]) * ϕ[i, j]
    end
    return s
end

@inline function mymaskedsum(
        f::F, A::AbstractArray, ϕ::AbstractArray, ranges_i, ranges_j, ranges_k
    ) where {F <: Function}
    s = 0.0
    for i in ranges_i, j in ranges_j, k in ranges_k
        s += f(A[i, j, k]) * ϕ[i, j, k]
    end
    return s
end
