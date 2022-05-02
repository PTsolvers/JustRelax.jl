export Geometry, IGG, lazy_grid, init_igg

struct Geometry{nDim}
    ni::NTuple{nDim,Integer}
    li::NTuple{nDim,Float64}
    max_li::Float64
    di::NTuple{nDim,Float64}
    xci::NTuple{nDim,StepRangeLen}
    xvi::NTuple{nDim,StepRangeLen}

    function Geometry(ni::NTuple{nDim,Integer}, li::NTuple{nDim,T}) where {nDim,T}
        li isa NTuple{nDim,Float64} == false && (li = Float64.(li))
        di = li ./ ni
        return new{nDim}(
            ni,
            li,
            Float64(max(li...)),
            di,
            Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]),
            Tuple([0:di[i]:li[i] for i in 1:nDim]),
        )
    end
end

struct IGG{T,M}
    me::T
    dims::Vector{T}
    nprocs::T
    coords::Vector{T}
    comm_cart::M
end

function lazy_grid(
    di::NTuple{N,T1}, li::NTuple{N,T2}; origin=ntuple(_ -> zero(T1), Val(N))
) where {N,T1,T2}
    @assert length(di) == length(li)
    nDim = Val(length(di))
    # nodes at the center of the grid cells
    xci = ntuple(
        i -> (origin[i] + di[i] * 0.5):di[i]:(origin[i] + li[i] - di[i] * 0.5), nDim
    )
    # nodes at the vertices of the grid cells
    xvi = ntuple(i -> origin[i]:di[i]:(origin[i] + li[i]), nDim)

    return xci, xvi
end
