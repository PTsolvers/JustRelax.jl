using MuladdMacro
abstract type AbstractMask end

struct Mask{T} <: AbstractMask
    mask::T

    Mask(A::T) where {T <: AbstractArray} = new{T}(A)
end

Mask(ni::Vararg{Int, N}) where {N} = Mask(zeros(ni...))

Mask(nx, ny, I, J) = Mask((nx, ny), (I, J))
Mask(nx, ny, nz, I, J, K) = Mask((nx, ny, nz), (I, J, K))

function Mask(ni::NTuple{N, Int}, inds::NTuple{N}) where {N}
    m = zeros(ni...)
    @views m[inds...] .= 1
    return Mask(m)
end

Adapt.@adapt_structure Mask

Base.@propagate_inbounds Base.getindex(m::Mask{T}, inds::Vararg{Int, N}) where {T, N} =
    m.mask[inds...]
Base.@propagate_inbounds Base.getindex(
    m::Mask{T}, inds::Vararg{UnitRange{Int64}, N}
) where {T, N} = m.mask[inds...]

Base.@propagate_inbounds Base.setindex!(m::Mask{T}, val, inds::Vararg{Int, N}) where {T, N} =
    setindex!(m.mask, val, inds...)
Base.@propagate_inbounds Base.setindex!(
    m::Mask{T}, val, inds::Vararg{UnitRange, N}
) where {T, N} = setindex!(m.mask, val, inds...)

Base.@propagate_inbounds Base.inv(m::Mask, inds::Vararg{Int, N}) where {N} = 1 - m[inds...]
Base.@propagate_inbounds Base.inv(m::Mask) = 1 .- m.mask

Base.size(m::Mask) = size(m.mask)
Base.length(m::Mask) = length(m.mask)
Base.axes(m::Mask) = axes(m.mask)
Base.eachindex(m::Mask) = eachindex(m.mask)
Base.all(m::Mask) = all(isone, m.mask)
Base.similar(m::Mask) = Mask(size(m)...)

@inline dims(::Mask{A}) where {A <: AbstractArray{T, N}} where {T, N} = N

@inline apply_mask!(A::AbstractArray, B::Any, m::Mask) = (A .= inv(m) .* A .+ m.mask .* B)
@inline apply_mask!(::AbstractArray, ::Any, ::Nothing) = nothing
@inline apply_mask!(A::AbstractArray, B::Any, m::Mask, inds::Vararg{Int, N}) where {N} =
    (A[inds...] = inv(m, inds...) * A[inds...] + m[inds...] * B[inds...])
@inline apply_mask!(::AbstractArray, ::Any, ::Nothing, inds::Vararg{Int, N}) where {N} =
    nothing

@inline apply_mask(A::AbstractArray, B::Any, m::Mask) = inv(m) .* A .+ m.mask .* B
@inline apply_mask(A::AbstractArray, B::Any, m::Mask, inds::Vararg{Int, N}) where {N} =
    @muladd inv(m, inds...) * A[inds...] + m[inds...] * B[inds...]
@inline apply_mask(A::AbstractArray, ::Any, ::Nothing) = A
@inline apply_mask(A::AbstractArray, ::Any, ::Nothing, inds::Vararg{Int, N}) where {N} =
    A[inds...]
