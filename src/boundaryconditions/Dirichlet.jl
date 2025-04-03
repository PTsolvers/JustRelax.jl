abstract type AbstractDirichletBoundaryCondition{T, M} end
struct DirichletBoundaryCondition{T, M} <: AbstractDirichletBoundaryCondition{T, M}
    value::T
    mask::M

    function DirichletBoundaryCondition(value::T, mask::M) where {T, M}
        return new{T, M}(value, mask)
    end
end

DirichletBoundaryCondition() = DirichletBoundaryCondition(nothing, nothing)

Adapt.@adapt_structure DirichletBoundaryCondition

function DirichletBoundaryCondition(A::AbstractArray{T}) where {T}
    m = Mask(copy(A))
    copyto!(m.mask, T.(.!iszero.(A)))

    return DirichletBoundaryCondition(A, m)
end

Base.getindex(x::DirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}) where {N} = 0
function Base.getindex(x::DirichletBoundaryCondition, inds::Vararg{Int, N}) where {N}
    return x.value[inds...] * x.mask[inds...]
end

struct ConstantArray{T}
    val::T

    ConstantArray(val::T) where {T <: Number} = new{T}(val)
end
Adapt.@adapt_structure ConstantArray

Base.getindex(A::ConstantArray, ::Vararg{Int, N}) where {N} = A.val
Base.setindex!(::ConstantArray, ::Any, ::Vararg{Int, N}) where {N} = nothing
function Base.show(io::IO, ::MIME"text/plain", A::ConstantArray{T}) where {T}
    println(io, "ConstantArray{$T}:")
    return println(io, "  ", A.val)
end

function Base.show(io::IO, A::ConstantArray{T}) where {T}
    println(io, "ConstantArray{$T}:")
    return println(io, "  ", A.val)
end

struct ConstantDirichletBoundaryCondition{T, M} <: AbstractDirichletBoundaryCondition{T, M}
    value::T
    mask::M
end

function ConstantDirichletBoundaryCondition(value::T, mask::M) where {T <: Number, M}
    v = ConstantArray(value)
    return ConstantDirichletBoundaryCondition{typeof(v), M}(v, mask)
end

function ConstantDirichletBoundaryCondition()
    return ConstantDirichletBoundaryCondition{Nothing, Nothing}(nothing, nothing)
end

Adapt.@adapt_structure ConstantDirichletBoundaryCondition

function Base.getindex(x::ConstantDirichletBoundaryCondition, inds::Vararg{Int, N}) where {N}
    return x.value * x.mask[inds...]
end
function Base.getindex(
        ::ConstantDirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}
    ) where {N}
    return 0
end

@inline function apply_dirichlet!(A::AbstractArray, bc::AbstractDirichletBoundaryCondition)
    apply_mask!(A, bc.value, bc.mask)
    return nothing
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
