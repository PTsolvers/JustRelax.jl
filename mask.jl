abstract type AbstractMask end

struct Mask{T} <: AbstractMask
    mask::T
    function Mask(ni::Vararg{Int,N}) where N
        mask = zeros(ni...)
        return new{typeof(mask)}(mask)
    end

    function Mask(nx, ny, inds::Vararg{UnitRange,N}) where N
        mask = zeros(nx, ny)
        @views mask[inds...] .= 1
        return new{typeof(mask)}(mask)
    end

    function Mask(nx, ny, nz, inds::Vararg{UnitRange,N}) where N
        mask = zeros(nx, ny, nz)
        @views mask[inds...] .= 1
        return new{typeof(mask)}(mask)
    end

end

Base.@propagate_inbounds Base.getindex(m::Mask{T}, inds::Vararg{Int,N}) where {T,N} = m.mask[inds...]
Base.@propagate_inbounds Base.getindex(m::Mask{T}, inds::Vararg{UnitRange{Int64},N}) where {T,N} = m.mask[inds...]

Base.@propagate_inbounds Base.setindex!(m::Mask{T}, val, inds::Vararg{Int,N}) where {T,N} = setindex!(m.mask, val, inds...)
Base.@propagate_inbounds Base.setindex!(m::Mask{T}, val, inds::Vararg{UnitRange,N}) where {T,N} = setindex!(m.mask, val, inds...)

Base.@propagate_inbounds Base.inv(m::Mask, inds::Vararg{Int,N}) where {N} =  1 - m[inds...]
Base.@propagate_inbounds Base.inv(m::Mask) =  1 .- m.mask

Base.size(m::Mask) = size(m.mask)

Base.length(m::Mask) = length(m.mask)

Base.axes(m::Mask) = axes(m.mask)

Base.eachindex(m::Mask) = eachindex(m.mask)

Base.all(m::Mask) = all(isone, m.mask)

Base.similar(m::Mask) = Mask(size(m)...)

apply_mask!(A::AbstractArray, B::AbstractArray, m::Mask) = (A .= inv(m) .* A .+ m.mask .* B)
apply_mask!(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = (A[inds...] = inv(m, inds...) * A[inds...] + m[inds...]  * B[inds...])

apply_mask(A::AbstractArray, B::AbstractArray, m::Mask) = inv(m) .* A .+ m.mask .* B
apply_mask(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = inv(m, inds...) * A[inds...] + m[inds...]  * B[inds...]