
@inline cellnum(A::CellArray) = prod(cellsize(A))
@inline cellaxes(A) = map(Base.oneto, cellnum(A))
@inline new_empty_cell(A::CellArray{T,N}) where {T,N} = zeros(T)

import Base.setindex!

Base.@propagate_inbounds @inline function Base.getindex(
    x::CellArray{SVector{Nv,T},N1,N2,T}, I::Vararg{Int,N}
) where {Nv,N,N1,N2,T}
    idx_cell = cart2ind(x.dims, I...)
    return SVector{Nv,T}(@inbounds x.data[idx_cell, i, 1] for i in 1:Nv)
end

Base.@propagate_inbounds @inline function Base.getindex(
    x::CPUCellArray{SVector{Nv,T},N1,N2,T}, I::Vararg{Int,N}
) where {Nv,N,N1,N2,T}
    idx_cell = cart2ind(x.dims, I...)
    return SVector{Nv,T}(@inbounds x.data[1, i, idx_cell] for i in 1:Nv)
end

Base.@propagate_inbounds @inline function setindex!(
    A::CellArray, x, cell::Int, I::Vararg{Int,N}
) where {N}
    Base.@propagate_inbounds @inline f(A::Array, x, cell, idx) = A[1, cell, idx] = x
    Base.@propagate_inbounds @inline f(A, x, cell, idx) = A[idx, cell, 1] = x
    
    idx = LinearIndices(n)[CartesianIndex(I...)]

    return f(A.data, x, cell, idx)
end

"""
    element(A, element_indices..., cell_indices...)

Return a the element with `element_indices` of the Cell with `cell_indices` of the CellArray `A`.

## Arguments
- `element_indices::Int|NTuple{N,Int}`: the `element_indices` that designate the field in accordance with `A`'s cell type.
- `cell_indices::Int|NTuple{N,Int}`: the `cell_indices` that designate the cell in accordance with `A`'s cell type.
"""
Base.@propagate_inbounds @inline function element(
    A::CellArray{SVector,N,D,T_elem}, i::Int, icell::Vararg{Int,Nc}
) where {T_elem,N,Nc,D}
    return viewelement(A, i, icell...)
end

Base.@propagate_inbounds @inline function element(
    A::CellArray, i::T, j::T, icell::Vararg{Int,Nc}
) where {Nc,T<:Int}
    return viewelement(A, i, j, icell...)
end

Base.@propagate_inbounds @inline function viewelement(
    A::CellArray{SMatrix{Ni,Nj,T,N_array},N,D,T_elem}, i, j, icell::Vararg{Int,Nc}
) where {Nc,Ni,Nj,N_array,T,N,T_elem,D}
    idx_element = cart2ind((Ni, Nj), i, j)
    idx_cell = cart2ind(A.dims, icell...)
    return _viewelement(A.data, idx_element, idx_cell)
end

Base.@propagate_inbounds @inline function viewelement(
    A::CellArray{SVector{Ni,T},N,D,T_elem}, i, icell::Vararg{Int,Nc}
) where {Nc,Ni,N,T,T_elem,D}
    idx_cell = cart2ind(A.dims, icell...)
    return _viewelement(A.data, i, idx_cell)
end

Base.@propagate_inbounds @inline _viewelement(A::Array, idx, icell) = A[1, idx, icell]
Base.@propagate_inbounds @inline _viewelement(A, idx, icell) = A[icell, idx, 1]

"""
    setelement!(A, x, element_indices..., cell_indices...)

Store the given value `x` at the given element with `element_indices` of the cell with the indices `cell_indices`

## Arguments
- `x::Number`: value to be stored in the index `element_indices` of the cell with `cell_indices`.
- `element_indices::Int|NTuple{N,Int}`: the `element_indices` that designate the field in accordance with `A`'s cell type.
- `cell_indices::Int|NTuple{N,Int}`: the `cell_indices` that designate the cell in accordance with `A`'s cell type.
"""
Base.@propagate_inbounds @inline function setelement!(
    A::CellArray{SMatrix{Ni,Nj,T,N_array},N,D,T_elem}, x::T, i, j, icell::Vararg{Int,Nc}
) where {Nc,Ni,Nj,N_array,T,N,T_elem,D}
    idx_element = cart2ind((Ni, Nj), i, j)
    idx_cell = cart2ind(A.dims, icell...)
    return _setelement!(A.data, x, idx_element, idx_cell)
end

Base.@propagate_inbounds @inline function setelement!(
    A::CellArray{SVector{Ni,T},N,D,T_elem}, x::T, i, icell::Vararg{Int,Nc}
) where {Nc,Ni,T,N,T_elem,D}
    idx_cell = cart2ind(A.dims, icell...)
    return _setelement!(A.data, x, i, idx_cell)
end

Base.@propagate_inbounds @inline function _setelement!(A::Array, x, idx::Int, icell::Int)
    return (A[1, idx, icell] = x)
end

Base.@propagate_inbounds @inline function _setelement!(A, x, idx::Int, icell::Int)
    return (A[icell, idx, 1] = x)
end

## Helper functions

"""
    cart2ind(A)

Return the linear index of a `n`-dimensional array corresponding to the cartesian indices `I`
"""
@inline function cart2ind(n::NTuple{N1,Int}, I::Vararg{Int,N2}) where {N1,N2}
    return LinearIndices(n)[CartesianIndex(I...)]
end
@inline cart2ind(ni::T, nj::T, i::T, j::T) where {T<:Int} = cart2ind((ni, nj), i, j)
@inline function cart2ind(ni::T, nj::T, nk::T, i::T, j::T, k::T) where {T<:Int}
    return cart2ind((ni, nj, nk), i, j, k)
end

## Convinience macros

macro cell(ex)
    ex = if ex.head === (:(=))
        _set(ex)
    else
        _get(ex)
    end
    return :($(esc(ex)))
end

@inline _get(ex) = Expr(:call, element, ex.args...)

@inline function _set(ex)
    return Expr(
        :call, setelement!, ex.args[1].args[1], ex.args[2], ex.args[1].args[2:end]...
    )
end
