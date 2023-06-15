## 2D mini kernels
const T2 = AbstractArray{<:Real, 2}

# finite differences
@inline Base.@propagate_inbounds _d_xa(A::T, i, j, _dx) where {T<:T2} = (A[i + 1, j    ] - A[i    , j    ]) * _dx
@inline Base.@propagate_inbounds _d_ya(A::T, i, j, _dy) where {T<:T2} = (A[i    , j + 1] - A[i    , j    ]) * _dy
@inline Base.@propagate_inbounds _d_xi(A::T, i, j, _dx) where {T<:T2} = (A[i + 1, j + 1] - A[i    , j + 1]) * _dx
@inline Base.@propagate_inbounds _d_yi(A::T, i, j, _dy) where {T<:T2} = (A[i + 1, j + 1] - A[i + 1, j    ]) * _dy
# averages
@inline Base.@propagate_inbounds _av(A::T, i, j) where {T<:T2} = (A[i + 1, j] + A[i + 2, j] + A[i + 1, j + 1] + A[i + 2, j + 1]) * 0.25
@inline Base.@propagate_inbounds _av_xa(A::T, i, j) where {T<:T2} = (A[i + 1, j    ] + A[i, j]) * 0.5
@inline Base.@propagate_inbounds _av_ya(A::T, i, j) where {T<:T2} = (A[i    , j + 1] + A[i, j]) * 0.5
#others
@inline Base.@propagate_inbounds _gather(A::T, i, j) where {T<:T2} = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]

## 3D mini kernels
const T3 = AbstractArray{<:Real, 3}

# averages
@inline Base.@propagate_inbounds _av(A::T, i, j, k) where {T<:T3} = 0.125 * (
    A[i, j, k    ] + A[i, j + 1, k    ] + A[i + 1, j, k    ] + A[i + 1, j + 1, k    ] +
    A[i, j, k + 1] + A[i, j + 1, k + 1] + A[i + 1, j, k + 1] + A[i + 1, j + 1, k + 1]
)
# others
@inline Base.@propagate_inbounds _gather_yz(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
@inline Base.@propagate_inbounds _gather_xz(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
@inline Base.@propagate_inbounds _gather_xy(A::T, i, j, k) where {T<:T3} = A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]

