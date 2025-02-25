abstract type AbstractDirichletBoundaryCondition{T, M} end
struct DirichletBoundaryCondition{T, M} <: AbstractDirichletBoundaryCondition{T, M}
    value::T
    mask::M

    function DirichletBoundaryCondition(value::T, mask::M) where {T, M}
        return new{T, M}(value, mask)
    end

    function DirichletBoundaryCondition()
        return new{Nothing, Nothing}(nothing, nothing)
    end
end

Adapt.@adapt_structure DirichletBoundaryCondition

function DirichletBoundaryCondition(A::AbstractArray{T}) where {T}
    m = Mask(size(A)...)
    copyto!(m.mask, T.(A .!= 0))
    return DirichletBoundaryCondition(A, m)
end

Base.getindex(x::DirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}) where {N} = 0
function Base.getindex(x::DirichletBoundaryCondition, inds::Vararg{Int, N}) where {N}
    return x.value[inds...] * x.mask[inds...]
end

struct ConstantArray{T, N} <: AbstractArray{T, N}
    val::T

    ConstantArray(val::T, ::Val{N}) where {T <: Number, N} = new{T, N}(val)
end
Adapt.@adapt_structure ConstantArray

Base.getindex(A::ConstantArray, ::Vararg{Int, N}) where {N} = A.val
Base.setindex!(::ConstantArray, ::Any, ::Vararg{Int, N}) where {N} = nothing
function Base.show(io::IO, ::MIME"text/plain", A::ConstantArray{T, N}) where {T, N}
    println(io, "ConstantArray{$T,$N}:")
    return println(io, "  ", A.val)
end

function Base.show(io::IO, A::ConstantArray{T, N}) where {T, N}
    println(io, "ConstantArray{$T,$N}:")
    return println(io, "  ", A.val)
end

struct ConstantDirichletBoundaryCondition{T, M} <: AbstractDirichletBoundaryCondition{T, M}
    value::T
    mask::M

    function ConstantDirichletBoundaryCondition(value::N, mask::M) where {N <: Number, M}
        v = ConstantArray(value, Val(dims(mask)))
        T = typeof(v)
        return new{T, M}(v, mask)
    end

    function ConstantDirichletBoundaryCondition()
        return new{Nothing, Nothing}(nothing, nothing)
    end
end

Adapt.@adapt_structure ConstantDirichletBoundaryCondition

function Base.getindex(x::ConstantDirichletBoundaryCondition, inds::Vararg{Int, N}) where {N}
    return x.value * x.mask[inds...]
end
function Base.getindex(
        x::ConstantDirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}
    ) where {N}
    return 0
end

@inline function apply_dirichlet!(A::AbstractArray, bc::AbstractDirichletBoundaryCondition)
    return apply_mask!(A, bc.value, bc.mask)
end

@inline function apply_dirichlet!(
        ::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}
    )
    return nothing
end

@inline function apply_dirichlet!(
        A::AbstractArray, bc::AbstractDirichletBoundaryCondition, inds::Vararg{Int, N}
    ) where {N}
    return apply_mask!(A, bc.value, bc.mask, inds...)
end

@inline function apply_dirichlet!(
        ::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}
    ) where {N}
    return nothing
end

@inline function apply_dirichlet(A::AbstractArray, bc::AbstractDirichletBoundaryCondition)
    return apply_mask(A, bc.value, bc.mask)
end

@inline function apply_dirichlet(
        A::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}
    )
    return A
end

@inline function apply_dirichlet(
        A::AbstractArray, bc::AbstractDirichletBoundaryCondition, inds::Vararg{Int, N}
    ) where {N}
    return apply_mask(A, bc.value, bc.mask, inds...)
end

@inline function apply_dirichlet(
        A::AbstractArray,
        ::AbstractDirichletBoundaryCondition{Nothing, Nothing},
        inds::Vararg{Int, N},
    ) where {N}
    return A[inds...]
end

@inline Dirichlet(x::NamedTuple) = Dirichlet(; x...)
@inline Dirichlet(; constant = nothing, mask = nothing) = Dirichlet(constant, mask)
@inline Dirichlet(::Nothing, mask::Nothing) = DirichletBoundaryCondition()
@inline Dirichlet(::Nothing, mask::AbstractArray) = DirichletBoundaryCondition(mask)
@inline Dirichlet(constant::Number, mask::AbstractArray) =
    ConstantDirichletBoundaryCondition(constant, Mask(mask))
