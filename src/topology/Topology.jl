# MPI topology

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

    function Geometry(ni::NTuple{nDim,Integer}, li::NTuple{nDim,T}, origin::NTuple{nDim,T} ) where {nDim,T}
        li isa NTuple{nDim,Float64} == false && (li = Float64.(li))
        di = li ./ ni
        xci, xvi = lazy_grid(di, li; origin=origin)

        return new{nDim}(
            ni,
            li,
            origin,
            Float64(max(li...)),
            di,
            xci,
            xvi,
        )
    end
end

function lazy_grid(
    di::NTuple{N,T1}, li::NTuple{N,T2}; origin=ntuple(_ -> zero(T1), Val(N))
) where {N,T1,T2}
    nDim = Val(N)
    # nodes at the center of the grid cells
    xci = ntuple(
        i -> (origin[i] + di[i] * 0.5):di[i]:(origin[i] + li[i] - di[i] * 0.5), nDim
    )
    # nodes at the vertices of the grid cells
    xvi = ntuple(i -> origin[i]:di[i]:(origin[i] + li[i]), nDim)

    return xci, xvi
end
