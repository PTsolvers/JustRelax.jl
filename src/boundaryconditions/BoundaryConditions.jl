@parallel_indices (iy) function free_slip_x!(A::AbstractArray{eltype(PTArray), 2}) 
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function free_slip_y!(A::AbstractArray{eltype(PTArray), 2})
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end
