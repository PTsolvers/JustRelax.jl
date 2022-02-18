function pureshear_bc!(stokes::StokesArrays, di::NTuple{2,T}, li::NTuple{2,T}, εbg) where T
    # unpack
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dx, dy = di 
    lx, ly = li 
    # Velocity boundary conditions
    stokes.V.Vx .= PTArray( -εbg.*[((ix-1)*dx -0.5*lx) for ix=1:size(Vx,1), iy=1:size(Vx,2)] )
    stokes.V.Vy .= PTArray(  εbg.*[((iy-1)*dy -0.5*ly) for ix=1:size(Vy,1), iy=1:size(Vy,2)] )
end

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
