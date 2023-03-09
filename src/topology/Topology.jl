# MPI struct

struct IGG{T,M}
    me::T
    dims::Vector{T}
    nprocs::T
    coords::Vector{T}
    comm_cart::M
end

# Staggered grid

struct Geometry{nDim}
    ni::NTuple{nDim,Integer}
    li::NTuple{nDim,Float64}
    origin::NTuple{nDim,Float64}
    max_li::Float64
    di::NTuple{nDim,Float64}
    xci::NTuple{nDim,StepRangeLen}
    xvi::NTuple{nDim,StepRangeLen}

    function Geometry(
        ni::NTuple{nDim,Integer}, li::NTuple{nDim,T}; origin=ntuple(_ -> 0.0, Val(nDim))
    ) where {nDim,T}
        Li = Float64.(li)
        di = Li ./ ni
        xci, xvi = lazy_grid(di, Li; origin=origin)
        return new{nDim}(ni, Li, max(Li...), di, xci, xvi)
    end
end

function lazy_grid(
    di::NTuple{N,T1}, li::NTuple{N,T2}; origin=ntuple(_ -> zero(T1), Val(N))
) where {N,T1,T2}
    # nodes at the center of the grid cells
    xci = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds (origin[i] + di[i] * 0.5):di[i]:(origin[i] + li[i] - di[i] * 0.5)
    end
    # nodes at the vertices of the grid cells
    xvi = ntuple(Val(N)) do i
        Base.@_inline_meta
        @inbounds origin[i]:di[i]:(origin[i] + li[i])
    end

    return xci, xvi
end
