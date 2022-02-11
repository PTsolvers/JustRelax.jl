struct Geometry{nDim}
    ni::NTuple{nDim, Integer}
    li::NTuple{nDim, Float64}
    max_li::Float64
    di::NTuple{nDim, Float64}
    xci::NTuple{nDim, StepRangeLen}
    xvi::NTuple{nDim, StepRangeLen}

    function Geometry(ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}) where {nDim, T}
        li isa NTuple{nDim, Float64} == false && (li = Float64.(li))
        di = li./ni
        new{nDim}(
            ni,
            li,
            Float64(max(li...)),
            di,
            Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]),
            Tuple([0:di[i]:li[i] for i in 1:nDim])
        )
    end

end