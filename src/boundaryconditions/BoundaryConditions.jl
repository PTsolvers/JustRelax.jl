# 2D KERNELS

function pureshear_bc!(
    stokes::StokesArrays, di::NTuple{2,T}, li::NTuple{2,T}, εbg
) where {T}
    # unpack
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dx, dy = di
    lx, ly = li
    # Velocity pure shear boundary conditions
    stokes.V.Vx .= PTArray([
        -εbg * ((ix - 1) * dx - 0.5 * lx) for ix in 1:size(Vx, 1), iy in 1:size(Vx, 2)
    ])
    return stokes.V.Vy .= PTArray([
        εbg * ((iy - 1) * dy - 0.5 * ly) for ix in 1:size(Vy, 1), iy in 1:size(Vy, 2)
    ])
end

@parallel_indices (iy) function free_slip_x!(A::AbstractArray{eltype(PTArray),2})
    A[1, iy] = A[2, iy]
    A[end, iy] = A[end - 1, iy]
    return nothing
end

@parallel_indices (ix) function free_slip_y!(A::AbstractArray{eltype(PTArray),2})
    A[ix, 1] = A[ix, 2]
    A[ix, end] = A[ix, end - 1]
    return nothing
end

@parallel_indices (ix) function zero_y!(A::AbstractArray{eltype(PTArray),2})
    A[ix, 1] = 0.0
    A[ix, end] = 0.0
    return nothing
end

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy) where {T}
    freeslip_x, freeslip_y = freeslip
    # free slip boundary conditions
    freeslip_x && (@parallel (1:size(Vy, 2)) free_slip_x!(Vy))

    freeslip_y && (@parallel (1:size(Vx, 1)) free_slip_y!(Vx))

    return nothing
end

function thermal_boundary_conditions!(
    insulation::NamedTuple, T::AbstractArray{_T,2}
) where {_T}
    insulation_x, insulation_y = insulation

    nx, ny = size(T)

    insulation_x && (@parallel (1:ny) free_slip_x!(T))

    insulation_y && (@parallel (1:nx) free_slip_y!(T))

    return nothing
end

# 3D KERNELS

@parallel_indices (iy, iz) function free_slip_x!(A::AbstractArray{T,3}) where {T}
    A[1, iy, iz] = A[2, iy, iz]
    A[end, iy, iz] = A[end - 1, iy, iz]
    return nothing
end

@parallel_indices (ix, iz) function free_slip_y!(A::AbstractArray{T,3}) where {T}
    A[ix, 1, iz] = A[ix, 2, iz]
    A[ix, end, iz] = A[ix, end - 1, iz]
    return nothing
end

@parallel_indices (ix, iy) function free_slip_z!(A::AbstractArray{T,3}) where {T}
    A[ix, iy, 1] = A[ix, iy, 2]
    A[ix, iy, end] = A[ix, iy, end - 1]
    return nothing
end

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{3,T}}, Vx, Vy, Vz) where {T}
    freeslip_x, freeslip_y, freeslip_z = freeslip
    # free slip boundary conditions
    if freeslip_x
        @parallel (1:size(Vy, 2), 1:size(Vy, 3)) free_slip_x!(Vy)
        @parallel (1:size(Vz, 2), 1:size(Vz, 3)) free_slip_x!(Vz)
    end
    if freeslip_y
        @parallel (1:size(Vx, 1), 1:size(Vx, 3)) free_slip_y!(Vx)
        @parallel (1:size(Vz, 1), 1:size(Vz, 3)) free_slip_y!(Vz)
    end
    if freeslip_z
        @parallel (1:size(Vx, 1), 1:size(Vx, 2)) free_slip_z!(Vx)
        @parallel (1:size(Vy, 1), 1:size(Vy, 2)) free_slip_z!(Vy)
    end
end

function thermal_boundary_conditions!(
    insulation::NamedTuple, T::AbstractArray{_T,3}
) where {_T}
    # vertical boundaries
    frontal, lateral = insulation

    nx, ny, nz = size(T)

    frontal && (@parallel (1:nx, 1:nz) free_slip_y!(T))
    lateral && (@parallel (1:ny, 1:nz) free_slip_x!(T))

    return nothing
end
