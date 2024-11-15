using MuladdMacro
abstract type AbstractMask end

struct Mask{T} <: AbstractMask
    mask::T

    Mask(A::T) where T<:AbstractArray = new{T}(A) 
end

Mask(ni::Vararg{Int,N}) where N = Mask(zeros(ni...))

function Mask(nx, ny, inds::Vararg{UnitRange,N}) where N
    mask = zeros(nx, ny)
    @views mask[inds...] .= 1
    return Mask(mask)
end

function Mask(nx, ny, nz, inds::Vararg{UnitRange,N}) where N
    mask = zeros(nx, ny, nz)
    @views mask[inds...] .= 1
    return Mask(mask)
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

@inline dims(::Mask{A}) where A<:AbstractArray{T,N} where {T,N} = N

@inline apply_mask!(A::AbstractArray, B::AbstractArray, m::Mask) = (A .= inv(m) .* A .+ m.mask .* B)
@inline apply_mask!(::AbstractArray, ::AbstractArray, ::Nothing) = nothing
@inline apply_mask!(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = (A[inds...] = inv(m, inds...) * A[inds...] + m[inds...]  * B[inds...])
@inline apply_mask!(::AbstractArray, ::AbstractArray, ::Nothing, inds::Vararg{Int,N}) where {N} = nothing

@inline apply_mask(A::AbstractArray, B::AbstractArray, m::Mask) = inv(m) .* A .+ m.mask .* B
@inline apply_mask(A::AbstractArray, B::AbstractArray, m::Mask, inds::Vararg{Int,N}) where {N} = @muladd inv(m, inds...) * A[inds...] + m[inds...] * B[inds...]
@inline apply_mask(A::AbstractArray, ::AbstractArray, ::Nothing) = A
@inline apply_mask(A::AbstractArray, ::AbstractArray, ::Nothing, inds::Vararg{Int,N}) where {N} = A[inds...]

## 
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

function DirichletBoundaryCondition(A::AbstractArray{T}) where T
    m = Mask(size(A)...) 
    copyto!(m.mask, T.(A .!= 0))
    return DirichletBoundaryCondition(A, m)
end

Base.getindex(x::DirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}) where N = 0
Base.getindex(x::DirichletBoundaryCondition, inds::Vararg{Int, N}) where N = x.value[inds...] * x.mask[inds...]

struct ConstantArray{T, N} <: AbstractArray{T, N}
    val::T

    ConstantArray(val::T, ::Val{N}) where {T<:Number,N} = new{T,N}(val)
end
Base.getindex(A::ConstantArray, ::Vararg{Int, N}) where N = A.val
Base.setindex!(::ConstantArray, ::Any, ::Vararg{Int, N}) where N = nothing
function Base.show(io::IO, ::MIME"text/plain", A::ConstantArray{T, N}) where {T, N}
    println(io, "ConstantArray{$T,$N}:")
    println(io, "  ", A.val)
end

function Base.show(io::IO, A::ConstantArray{T, N}) where {T, N}
    println(io, "ConstantArray{$T,$N}:")
    println(io, "  ", A.val)
end

struct ConstantDirichletBoundaryCondition{T, M} <: AbstractDirichletBoundaryCondition{T, M}
    value::T
    mask::M

    function ConstantDirichletBoundaryCondition(value::N, mask::M) where {N<:Number, M}
        v = ConstantArray(value, Val(dims(mask)))
        T = typeof(v)
        return new{T, M}(v, mask)
    end

    function ConstantDirichletBoundaryCondition()
        return new{Nothing, Nothing}(nothing, nothing)
    end
end

Base.getindex(x::ConstantDirichletBoundaryCondition, inds::Vararg{Int, N}) where N = x.value * x.mask[inds...]
Base.getindex(x::ConstantDirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int, N}) where N = 0

@inline apply_dirichlet!(A::AbstractArray, bc::AbstractDirichletBoundaryCondition) = apply_mask!(A, bc.value, bc.mask)
@inline apply_dirichlet!(::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}) = nothing
@inline apply_dirichlet!(A::AbstractArray, bc::AbstractDirichletBoundaryCondition, inds::Vararg{Int,N}) where {N} = apply_mask!(A, bc.value, bc.mask, inds...)
@inline apply_dirichlet!(::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}, ::Vararg{Int,N}) where {N} = nothing

@inline apply_dirichlet(A::AbstractArray, bc::AbstractDirichletBoundaryCondition) = apply_mask(A, bc.value, bc.mask)
@inline apply_dirichlet(A::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}) = A
@inline apply_dirichlet(A::AbstractArray, bc::AbstractDirichletBoundaryCondition, inds::Vararg{Int,N}) where {N} = apply_mask(A, bc.value, bc.mask, inds...)
@inline apply_dirichlet(A::AbstractArray, ::AbstractDirichletBoundaryCondition{Nothing, Nothing}, inds::Vararg{Int,N}) where {N} = A[inds...]

@inline Dirichlet(x::NamedTuple) = Dirichlet(; x...)
@inline Dirichlet(; values = nothing, mask = nothing) = Dirichlet(values, mask)
@inline Dirichlet(::Nothing, mask::Nothing) = DirichletBoundaryCondition()
@inline Dirichlet(::Nothing, mask::AbstractArray) = DirichletBoundaryCondition(mask)
@inline Dirichlet(values::Number, mask::AbstractArray) = ConstantDirichletBoundaryCondition(values, Mask(mask))

