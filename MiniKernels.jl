# unmasked versions
Base.@propagate_inbounds @inline center(A::AbstractArray, inds::Vararg{Integer, N}) where N = A[inds...]
Base.@propagate_inbounds @inline  next(A::AbstractArray, inds::Vararg{Integer, N}) where N  = A[inds.+1...]
Base.@propagate_inbounds @inline  left(A::AbstractArray, i::I, j::I) where I<:Integer = A[i-1, j]
Base.@propagate_inbounds @inline right(A::AbstractArray, i::I, j::I) where I<:Integer = A[i+1, j]
Base.@propagate_inbounds @inline  back(A::AbstractArray, i::I, j::I) where I<:Integer = A[i, j-1]
Base.@propagate_inbounds @inline front(A::AbstractArray, i::I, j::I) where I<:Integer = A[i, j+1]
Base.@propagate_inbounds @inline  left(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i-1, j, k]
Base.@propagate_inbounds @inline right(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i+1, j, k]
Base.@propagate_inbounds @inline  back(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i, j-1, k]
Base.@propagate_inbounds @inline front(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i, j+1, k]
Base.@propagate_inbounds @inline   bot(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i, j, k-1]
Base.@propagate_inbounds @inline   top(A::AbstractArray, i::I, j::I, k::I) where I<:Integer = A[i, j, k+1]

# masked versions
for fn in (:center, :next, :left, :right, :back, :front)
    @eval begin
        Base.@propagate_inbounds @inline ($fn)(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = ($fn)(A, inds...) * ($fn)(ϕ, inds...)        
    end
end
# Base.@propagate_inbounds @inline center(A::T,ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = center(A, inds...) * center(ϕ, inds...)
# Base.@propagate_inbounds @inline  next(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = next(A, inds...)   * next(ϕ, inds...)
# Base.@propagate_inbounds @inline  left(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = left(A, inds...)   * left(ϕ, inds...)
# Base.@propagate_inbounds @inline right(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = right(A, inds...)  * right(ϕ, inds...)
# Base.@propagate_inbounds @inline  back(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = back(A, inds...)   * back(ϕ, inds...)
# Base.@propagate_inbounds @inline front(A::T, ϕ::T, inds::Vararg{Integer, N}) where {T<:AbstractArray, N} = front(A, inds...)  * front(ϕ, inds...)

## 2D mini kernels
const T2 = AbstractArray{T,2} where {T}

# finite differences
Base.@propagate_inbounds @inline _d_xa(A::AbstractArray, _dx, I::Vararg{Integer, N}) where {N} = (-center(A, I...) + right(A, I...)) * _dx
Base.@propagate_inbounds @inline _d_ya(A::AbstractArray, _dy, I::Vararg{Integer, N}) where {N} = (-center(A, I...) + front(A, I...)) * _dy
Base.@propagate_inbounds @inline _d_za(A::AbstractArray, _dz, I::Vararg{Integer, N}) where {N} = (-center(A, I...) + top(A, I...)) * _dz
Base.@propagate_inbounds @inline _d_xi(A::AbstractArray, _dx, I::Vararg{Integer, N}) where {N} = (-front(A, I...) + next(A, I...)) * _dx
Base.@propagate_inbounds @inline _d_yi(A::AbstractArray, _dy, I::Vararg{Integer, N}) where {N} = (-right(A, I...) + next(A, I...)) * _dy
Base.@propagate_inbounds @inline _d_xa(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + right(A, ϕ, I...)) * _dx
Base.@propagate_inbounds @inline _d_ya(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dy
Base.@propagate_inbounds @inline _d_za(A::T, ϕ::T, _dz, I::Vararg{Integer, N}) where {N, T} = (-center(A, ϕ, I...) + front(A, ϕ, I...)) * _dz
Base.@propagate_inbounds @inline _d_xi(A::T, ϕ::T, _dx, I::Vararg{Integer, N}) where {N, T} = (-front(A, ϕ, I...) + next(A, ϕ, I...)) * _dx
Base.@propagate_inbounds @inline _d_yi(A::T, ϕ::T, _dy, I::Vararg{Integer, N}) where {N, T} = (-right(A, ϕ, I...) + next(A, ϕ, I...)) * _dy

# averages
Base.@propagate_inbounds @inline _av(A::T, i, j) where {T<:T2} = 0.25 * mysum(A, (i + 1):(i + 2), (j + 1):(j + 2))
Base.@propagate_inbounds @inline _av_a(A::T, i, j) where {T<:T2} = 0.25 * mysum(A, (i):(i + 1), (j):(j + 1))
Base.@propagate_inbounds @inline _av_xa(A::T, i, j) where {T<:T2} = (center(A, i, j) + right(A, i, j)) * 0.5
Base.@propagate_inbounds @inline _av_ya(A::T, i, j) where {T<:T2} = (center(A, i, j) + front(A, i, j)) * 0.5
Base.@propagate_inbounds @inline _av_xi(A::T, i, j) where {T<:T2} = (front(A, i, j), next(A, i, j)) * 0.5
Base.@propagate_inbounds @inline _av_yi(A::T, i, j) where {T<:T2} = (right(A, i, j), next(A, i, j)) * 0.5
# harmonic averages
Base.@propagate_inbounds @inline function _harm(A::T, i, j) where {T<:T2}
    return eltype(A)(4) * mysum(inv, A, (i + 1):(i + 2), (j + 1):(j + 2))
end
Base.@propagate_inbounds @inline function _harm_a(A::T, i, j) where {T<:T2}
    return eltype(A)(4) * mysum(inv, A, (i):(i + 1), (j):(j + 1))
end
Base.@propagate_inbounds @inline function _harm_xa(A::T, i, j) where {T<:T2}
    return eltype(A)(2) * (inv(right(A, i, j)) + inv(center(A, i, j)))
end
Base.@propagate_inbounds @inline function _harm_ya(A::T, i, j) where {T<:T2}
    return eltype(A)(2) * (inv(front(A, i, j)) + inv(center(A, i, j)))
end
#others
Base.@propagate_inbounds @inline function _gather(A::T, i, j) where {T<:T2}
    return center(A, i, j), right(A, i, j), front(A, i, j), next(A, i, j)
end

## 3D mini kernels
const T3 = AbstractArray{T,3} where {T}

# finite differences
Base.@propagate_inbounds @inline function _d_xi(A::T, i, j, k, _dx) where {T<:T3}
    return (-A[i, j + 1, k + 1] + next(A, i, j, k)) * _dx
end
Base.@propagate_inbounds @inline function _d_yi(A::T, i, j, k, _dy) where {T<:T3}
    return (-A[i + 1, j, k + 1] + next(A, i, j, k)) * _dy
end
Base.@propagate_inbounds @inline function _d_zi(A::T, i, j, k, _dz) where {T<:T3}
    return (-A[i + 1, j + 1, k] + next(A, i, j, k)) * _dz
end
# averages
Base.@propagate_inbounds @inline _av(A::T, i, j, k) where {T<:T3} = 0.125 * mysum(A, i:(i + 1), j:(j + 1), k:(k + 1))
Base.@propagate_inbounds @inline _av_x(A::T, i, j, k) where {T<:T3} = 0.5 * (center(A, i, j, k) + right(A, i, j, k))
Base.@propagate_inbounds @inline _av_y(A::T, i, j, k) where {T<:T3} = 0.5 * (center(A, i, j, k) + front(A, i, j, k))
Base.@propagate_inbounds @inline _av_z(A::T, i, j, k) where {T<:T3} = 0.5 * (center(A, i, j, k) + top(A, i, j, k))
Base.@propagate_inbounds @inline _av_xy(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, i:(i + 1), j:(j + 1), k:k)
Base.@propagate_inbounds @inline _av_xz(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, i:(i + 1), j:j, k:(k + 1))
Base.@propagate_inbounds @inline _av_yz(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, i:i, j:(j + 1), k:(k + 1))
Base.@propagate_inbounds @inline _av_xyi(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, (i - 1):i, (j - 1):j, k:k)
Base.@propagate_inbounds @inline _av_xzi(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, (i - 1):i, j:j, (k - 1):k)
Base.@propagate_inbounds @inline _av_yzi(A::T, i, j, k) where {T<:T3} = 0.25 * mysum(A, i:i, (j - 1):j, (k - 1):k)
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
@inline function _harm_xy(A::T, i, j, k) where {T<:T3}
    return eltype(A)(4) * inv(mysum(A, i:(i + 1), j:(j + 1), k:k))
end
@inline function _harm_xz(A::T, i, j, k) where {T<:T3}
    return eltype(A)(4) * inv(mysum(A, i:(i + 1), j:j, k:(k + 1)))
end
@inline function _harm_yz(A::T, i, j, k) where {T<:T3}
    return eltype(A)(4) * inv(mysum(A, i:i, j:(j + 1), k:(k + 1)))
end
@inline function _harm_xyi(A::T, i, j, k) where {T<:T3}
    return eltype(A)(4) * inv(mysum(A, (i - 1):i, (j - 1):j, k:k))
end
@inline function _harm_xzi(A::T, i, j, k) where {T<:T3}
    return eltype(A)(4) * inv(mysum(A, (i - 1):i, j:j, (k - 1):k))
end
@inline function _harm_yzi(A::T, i, j, k) where {T<:T3}
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
@inline mysum(A, ranges::Vararg{T,N}) where {T,N} = mysum(identity, A, ranges...)

@inline function mysum(f::F, A, ranges_i) where {F<:Function}
    s = 0.0
    for i in ranges_i
        s += f(A[i])
    end
    return s
end

@inline function mysum(f::F, A, ranges_i, ranges_j) where {F<:Function}
    s = 0.0
    for i in ranges_i, j in ranges_j
        s += f(center(A, i, j))
    end
    return s
end

@inline function mysum(f::F, A, ranges_i, ranges_j, ranges_k) where {F<:Function}
    s = 0.0
    for i in ranges_i, j in ranges_j, k in ranges_k
        s += f(center(A, i, j, k))
    end
    return s
end
