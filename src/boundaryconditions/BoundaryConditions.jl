# 2D KERNELS

@parallel_indices (iy) function free_slip_x!(A::AbstractArray{eltype(PTArray), 2}) <
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function free_slip_y!(A::AbstractArray{eltype(PTArray), 2})
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

# 3D KERNELS

@parallel_indices (iy, iz) function free_slip_x!(A::AbstractArray{eltype(PTArray), 3}) 
    A[1  , iy, iz] = A[2    , iy, iz]
    A[end, iy, iz] = A[end-1, iy, iz]
    return
end

@parallel_indices (ix, iz) function free_slip_y!(A::AbstractArray{eltype(PTArray), 3})
    A[ix, 1  , iz] = A[ix, 2    , iz]
    A[ix, end, iz] = A[ix, end-1, iz]
    return
end

@parallel_indices (ix, iy) function free_slip_y!(A::AbstractArray{eltype(PTArray), 3})
    A[ix, iy, 1  ] = A[ix, iy, 2    ]
    A[ix, iy, end] = A[ix, iy, end-1]
    return
end
