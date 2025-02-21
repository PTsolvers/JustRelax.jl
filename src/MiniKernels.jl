# unmasked versions
Base.@propagate_inbounds @inline center(
    A::AbstractArray, inds::Vararg{Integer,N}
) where {N} = A[inds...]
Base.@propagate_inbounds @inline next(A::AbstractArray, inds::Vararg{Integer,N}) where {N} =
    A[inds .+ 1...]
Base.@propagate_inbounds @inline left(A::AbstractArray, i::I, j::I) where {I<:Integer} =
    A[i - 1, j]
Base.@propagate_inbounds @inline right(A::AbstractArray, i::I, j::I) where {I<:Integer} =
    A[i + 1, j]
Base.@propagate_inbounds @inline back(A::AbstractArray, i::I, j::I) where {I<:Integer} =
    A[i, j - 1]
Base.@propagate_inbounds @inline front(A::AbstractArray, i::I, j::I) where {I<:Integer} =
    A[i, j + 1]
Base.@propagate_inbounds @inline left(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i - 1, j, k]
Base.@propagate_inbounds @inline right(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i + 1, j, k]
Base.@propagate_inbounds @inline back(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i, j - 1, k]
Base.@propagate_inbounds @inline front(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i, j + 1, k]
Base.@propagate_inbounds @inline bot(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i, j, k - 1]
Base.@propagate_inbounds @inline top(
    A::AbstractArray, i::I, j::I, k::I
) where {I<:Integer} = A[i, j, k + 1]

## 2D mini kernels
const T2 = AbstractArray{T, 2} where {T}

# finite differences
Base.@propagate_inbounds @inline _d_xa(
    A::AbstractArray, _dx, I::Vararg{Integer,N}
) where {N} = (-center(A, I...) + right(A, I...)) * _dx
Base.@propagate_inbounds @inline _d_ya(
    A::AbstractArray, _dy, I::Vararg{Integer,N}
) where {N} = (-center(A, I...) + front(A, I...)) * _dy
Base.@propagate_inbounds @inline _d_za(
    A::AbstractArray, _dz, I::Vararg{Integer,N}
) where {N} = (-center(A, I...) + top(A, I...)) * _dz
Base.@propagate_inbounds @inline _d_xi(
    A::AbstractArray, _dx, I::Vararg{Integer,N}
) where {N} = (-front(A, I...) + next(A, I...)) * _dx
Base.@propagate_inbounds @inline _d_yi(
    A::AbstractArray, _dy, I::Vararg{Integer,N}
) where {N} = (-right(A, I...) + next(A, I...)) * _dy

Base.@propagate_inbounds @inline div(Ax, Ay, _dx, _dy, I::Vararg{Integer,2}) =
    _d_xi(Ax, _dx, I...) + _d_yi(Ay, _dy, I...)

# averages
Base.@propagate_inbounds @inline _av(A::T, i, j) where {T<:T2} =
    0.25 * mysum(A, (i + 1):(i + 2), (j + 1):(j + 2))
Base.@propagate_inbounds @inline _av_a(A::T, i, j) where {T<:T2} =
    0.25 * mysum(A, (i):(i + 1), (j):(j + 1))
Base.@propagate_inbounds @inline _av_xa(A::T, I::Vararg{Integer,2}) where {T<:T2} =
    (center(A, I...) + right(A, I...)) * 0.5
Base.@propagate_inbounds @inline _av_ya(A::T, I::Vararg{Integer,2}) where {T<:T2} =
    (center(A, I...) + front(A, I...)) * 0.5
Base.@propagate_inbounds @inline _av_xi(A::T, I::Vararg{Integer,2}) where {T<:T2} =
    (front(A, I...), next(A, I...)) * 0.5
Base.@propagate_inbounds @inline _av_yi(A::T, I::Vararg{Integer,2}) where {T<:T2} =
    (right(A, I...), next(A, I...)) * 0.5
# harmonic averages
Base.@propagate_inbounds @inline function _harm(A::T, i, j) where {T<:T2}
    return eltype(A)(4) * mysum(inv, A, (i + 1):(i + 2), (j + 1):(j + 2))
end
Base.@propagate_inbounds @inline function _harm_a(A::T, i, j) where {T<:T2}
    return eltype(A)(4) * mysum(inv, A, (i):(i + 1), (j):(j + 1))
end
Base.@propagate_inbounds @inline function _harm_xa(A::T, I::Vararg{Integer,2}) where {T<:T2}
    return eltype(A)(2) * (inv(right(A, I...)) + inv(center(A, I...)))
end
Base.@propagate_inbounds @inline function _harm_ya(A::T, I::Vararg{Integer,2}) where {T<:T2}
    return eltype(A)(2) * (inv(front(A, I...)) + inv(center(A, I...)))
end
#others
Base.@propagate_inbounds @inline function _gather(A::T, I::Vararg{Integer,2}) where {T<:T2}
    return center(A, I...), right(A, I...), front(A, I...), next(A, I...)
end

## 3D mini kernels
const T3 = AbstractArray{T, 3} where {T}

# finite differences
@inline function _d_zi(A::T, i, j, k, _dz) where {T<:T3}
    return (-A[i + 1, j + 1, k] + next(A, i, j, k)) * _dz
end
Base.@propagate_inbounds @inline div(Ax, Ay, Az, _dx, _dy, _dz, I::Vararg{Integer,3}) =
    _d_xi(Ax, _dx, I...) + _d_yi(Ay, _dy, I...) + _d_zi(Az, _dz, I...)

# averages
Base.@propagate_inbounds @inline _av(A::T, i, j, k) where {T<:T3} =
    0.125 * mysum(A, i:(i + 1), j:(j + 1), k:(k + 1))
Base.@propagate_inbounds @inline _av_x(A::T, i, j, k) where {T<:T3} =
    0.5 * (center(A, i, j, k) + right(A, i, j, k))
Base.@propagate_inbounds @inline _av_y(A::T, i, j, k) where {T<:T3} =
    0.5 * (center(A, i, j, k) + front(A, i, j, k))
Base.@propagate_inbounds @inline _av_z(A::T, i, j, k) where {T<:T3} =
    0.5 * (center(A, i, j, k) + top(A, i, j, k))
Base.@propagate_inbounds @inline _av_xy(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, i:(i + 1), j:(j + 1), k:k)
Base.@propagate_inbounds @inline _av_xz(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, i:(i + 1), j:j, k:(k + 1))
Base.@propagate_inbounds @inline _av_yz(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, i:i, j:(j + 1), k:(k + 1))
Base.@propagate_inbounds @inline _av_xyi(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, (i - 1):i, (j - 1):j, k:k)
Base.@propagate_inbounds @inline _av_xzi(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, (i - 1):i, j:j, (k - 1):k)
Base.@propagate_inbounds @inline _av_yzi(A::T, i, j, k) where {T<:T3} =
    0.25 * mysum(A, i:i, (j - 1):j, (k - 1):k)
# harmonic averages
@inline function _harm_x(A::T, i, j, k) where {T<:T3}
    return eltype(A)(2) * inv(inv(center(A, i, j, k)) + inv(right(A, i, j, k)))
end
@inline function _harm_y(A::T, i, j, k) where {T<:T3}
    return eltype(A)(2) * inv(inv(center(A, i, j, k)) + inv(front(A, i, j, k)))
end
@inline function _harm_z(A::T, i, j, k) where {T<:T3}
    return eltype(A)(2) * inv(inv(center(A, i, j, k)) + inv(top(A, i, j, k)))
end
@inline function _harm_xy(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, i:(i + 1), j:(j + 1), k:k))
end
@inline function _harm_xz(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, i:(i + 1), j:j, k:(k + 1)))
end
@inline function _harm_yz(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, i:i, j:(j + 1), k:(k + 1)))
end
@inline function _harm_xyi(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, (i - 1):i, (j - 1):j, k:k))
end
@inline function _harm_xzi(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, (i - 1):i, j:j, (k - 1):k))
end
@inline function _harm_yzi(A::T, i, j, k) where {T <: T3}
    return eltype(A)(4) * inv(mysum(A, i:i, (j - 1):j, (k - 1):k))
end

# others
Base.@propagate_inbounds @inline function _gather_yz(A::T, i, j, k) where {T<:T3}
    return center(A, i, j, k), front(A, i, j, k), top(A, i, j, k), A[i, j + 1, k + 1]
end
Base.@propagate_inbounds @inline function _gather_xz(A::T, i, j, k) where {T<:T3}
    return center(A, i, j, k), right(A, i, j, k), top(A, i, j, k), A[i + 1, j, k + 1]
end
Base.@propagate_inbounds @inline function _gather_xy(A::T, i, j, k) where {T<:T3}
    return center(A, i, j, k), right(A, i, j, k), front(A, i, j, k), A[i + 1, j + 1, k]
end
@inline _current(A::T, i, j, k) where {T<:T3} = center(A, i, j, k)

## Because mysum(::generator) does not work inside CUDA kernels...
@inline mysum(A, ranges::Vararg{T, N}) where {T, N} = mysum(identity, A, ranges...)

@inline function mysum(f::F, A::AbstractArray, ranges_i) where {F<:Function}
    s = 0.0
    for i in ranges_i
        s += f(A[i])
    end
    return s
end

@inline function mysum(f::F, A::AbstractArray, ranges_i, ranges_j) where {F<:Function}
    s = 0.0
    for i in ranges_i, j in ranges_j
        s += f(A[i, j])
    end
    return s
end

@inline function mysum(
    f::F, A::AbstractArray, ranges_i, ranges_j, ranges_k
) where {F<:Function}
    s = 0.0
    for i in ranges_i, j in ranges_j, k in ranges_k
        s += f(A[i, j, k])
    end
    return s
end
