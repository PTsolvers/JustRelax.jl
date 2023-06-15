## 2D mini kernels
const T2 = AbstractArray{<:Real, 2}

# finite differences
@inline _d_xa(A::T, i, j, _dx) where {T<:T2} = (A[i + 1, j    ] - A[i    , j    ]) * _dx
@inline _d_ya(A::T, i, j, _dy) where {T<:T2} = (A[i    , j + 1] - A[i    , j    ]) * _dy
@inline _d_xi(A::T, i, j, _dx) where {T<:T2} = (A[i + 1, j + 1] - A[i    , j + 1]) * _dx
@inline _d_yi(A::T, i, j, _dy) where {T<:T2} = (A[i + 1, j + 1] - A[i + 1, j    ]) * _dy
# averages
@inline    _av(A::T, i, j) where {T<:T2} = 0.25 * sum(A[ii, jj] for ii in i+1:i+2, jj in j+1:j+2)
@inline _av_xa(A::T, i, j) where {T<:T2} = (A[i + 1, j    ] + A[i, j]) * 0.5
@inline _av_ya(A::T, i, j) where {T<:T2} = (A[i    , j + 1] + A[i, j]) * 0.5
# harmonic averages
@inline    _harm(A::T, i, j) where {T<:T2} = eltype(A)(4) * sum(inv(A[ii, jj]) for ii in i+1:i+2, jj in j+1:j+2)
@inline _harm_xa(A::T, i, j) where {T<:T2} = eltype(A)(2) * (inv(A[i + 1, j    ]) + inv(A[i, j]))
@inline _harm_ya(A::T, i, j) where {T<:T2} = eltype(A)(2) * (inv(A[i    , j + 1]) + inv(A[i, j]))
#others
@inline _gather(A::T, i, j) where {T<:T2} = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]

## 3D mini kernels
const T3 = AbstractArray{<:Real, 3}

# finite differences
@inline  _dx(A::T, i, j, k)   where {T<:T3} = A[i + 1, j, k] - A[i, j, k]
@inline  _dy(A::T, i, j, k)   where {T<:T3} = A[i, j + 1, k] - A[i, j, k]
@inline  _dz(A::T, i, j, k)   where {T<:T3} = A[i, j, k + 1] - A[i, j, k]
@inline  _d_xi(A::T, i, j, k) where {T<:T3} = A[i + 1, j + 1, k + 1] - A[i    , j + 1, k + 1]
@inline  _d_yi(A::T, i, j, k) where {T<:T3} = A[i + 1, j + 1, k + 1] - A[i + 1, j    , k + 1]
@inline  _d_zi(A::T, i, j, k) where {T<:T3} = A[i + 1, j + 1, k + 1] - A[i + 1, j + 1, k    ]
# averages
@inline    _av(A::T, i, j, k) where {T<:T3} = 0.125 * sum(A[ii, jj, kk] for ii in i:i+1, jj in j:j+1, kk in k:k+1)
@inline  _av_x(A::T, i, j, k) where {T<:T3} = 0.5 * (A[i + 1, j, k] + A[i, j, k])
@inline  _av_y(A::T, i, j, k) where {T<:T3} = 0.5 * (A[i, j + 1, k] + A[i, j, k])
@inline  _av_z(A::T, i, j, k) where {T<:T3} = 0.5 * (A[i, j, k + 1] + A[i, j, k])
@inline _av_xy(A::T, i, j, k) where {T<:T3} = 0.25 * sum(A[ii, jj, k] for ii in i:i-1, jj in j:j-1)
@inline _av_xz(A::T, i, j, k) where {T<:T3} = 0.25 * sum(A[ii, j, kk] for ii in i:i-1, kk in k:k-1)
@inline _av_yz(A::T, i, j, k) where {T<:T3} = 0.25 * sum(A[i, jj, kk] for jj in j:j-1, kk in k:k-1)
# harmonic averages
@inline  _harm_x(A::T, i, j, k) where {T<:T3} = eltype(A)(2) * inv(inv(A[i + 1, j, k]) + inv(A[i, j, k]))
@inline  _harm_y(A::T, i, j, k) where {T<:T3} = eltype(A)(2) * inv(inv(A[i, j + 1, k]) + inv(A[i, j, k]))
@inline  _harm_z(A::T, i, j, k) where {T<:T3} = eltype(A)(2) * inv(inv(A[i, j, k + 1]) + inv(A[i, j, k]))
@inline _harm_xy(A::T, i, j, k) where {T<:T3} = eltype(A)(4) * inv(sum(inv(A[ii, jj, k]) for ii in i:i-1, jj in j:j-1))
@inline _harm_xz(A::T, i, j, k) where {T<:T3} = eltype(A)(4) * inv(sum(inv(A[ii, j, kk]) for ii in i:i-1, kk in k:k-1))
@inline _harm_yz(A::T, i, j, k) where {T<:T3} = eltype(A)(4) * inv(sum(inv(A[i, jj, kk]) for jj in j:j-1, kk in k:k-1))
# others
@inline _gather_yz(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
@inline _gather_xz(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
@inline _gather_xy(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]
@inline _current(A::T, i, j, k)   where {T<:T3} = A[i, j, k]