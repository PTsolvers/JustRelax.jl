using MuladdMacro
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
apply_mask!(::AbstractArray, ::AbstractArray, ::Nothing) = nothing
apply_mask!(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = (A[inds...] = inv(m, inds...) * A[inds...] + m[inds...]  * B[inds...])
apply_mask!(::AbstractArray, ::AbstractArray, ::Nothing, inds::Vararg{Int,N}) where {N} = nothing

apply_mask(A::AbstractArray, B::AbstractArray, m::Mask) = inv(m) .* A .+ m.mask .* B
apply_mask(::AbstractArray, ::AbstractArray, ::Nothing) = A
apply_mask(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = @muladd inv(m, inds...) * A[inds...] + m[inds...] * B[inds...]
apply_mask(::AbstractArray, ::AbstractArray, ::Nothing, inds::Vararg{Int,N}) where {N} = A[inds...]

## 
struct InnerDirichletBoundaryConditions{N, A, M}
    values::NTuple{N, A}
    masks::NTuple{N, M}

    function InnerDirichletBoundaryConditions(values::NTuple{N, A}, masks::NTuple{N, M}) where {N, A, M}
        return new{N, A, M}(values, masks)
    end

    function InnerDirichletBoundaryConditions(values::Tuple{}, masks::Tuple{N})
        return new{0, Nothing, Nothing}(values, masks)
    end
end

InnerDirichletBoundaryConditions() = InnerDirichletBoundaryConditions((), ())


struct InnerDirichletBoundaryCondition{A, M}
    values::A
    masks::M

    function InnerDirichletBoundaryCondition(value::A, mask::M) where {A, M}
        return new{A, M}(value, mask)
    end

    function InnerDirichletBoundaryCondition()
        return new{Nothing, Nothing}(nothing, nothing)
    end
end


value  = zeros(ni...)
value[4:7, 4:7] .= 5
mask  = Mask(ni..., 4:7, 4:7)

InnerDirichletBoundaryCondition(value, mask) 

@test InnerDirichletBoundaryCondition() isa InnerDirichletBoundaryCondition{Nothing, Nothing}
