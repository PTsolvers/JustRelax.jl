# 2D KERNELS

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

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy) where {T}
    freeslip_x, freeslip_y, freeslip_z = freeslip
    # free slip boundary conditions
    if freeslip_x
        @parallel (1:size(Vy, 2)) free_slip_x!(Vy)
    end
    if freeslip_y
        @parallel (1:size(Vx, 1)) free_slip_y!(Vx)
    end
end

# 3D KERNELS

@parallel_indices (iy, iz) function free_slip_x!(A::AbstractArray{eltype(PTArray),3})
    A[1, iy, iz] = A[2, iy, iz]
    A[end, iy, iz] = A[end - 1, iy, iz]
    return nothing
end

@parallel_indices (ix, iz) function free_slip_y!(A::AbstractArray{eltype(PTArray),3})
    A[ix, 1, iz] = A[ix, 2, iz]
    A[ix, end, iz] = A[ix, end - 1, iz]
    return nothing
end

@parallel_indices (ix, iy) function free_slip_z!(A::AbstractArray{eltype(PTArray),3})
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
    bcs::NamedTuple{<:Any,NTuple{3,_T}}, T; corners=(false, false, false)
) where {_T}
    bc_x, bc_y, bc_z = bcs
    nx, ny, nz = size(T)
    corner_x, corner_y, corner_z = ntuple(Val(3)) do i
        corners[i] == true ? (1, 1) : (0, 0)
    end
    # flux boundary conditions
    if bc_x
        @parallel ((2 - corner_y[1]):(ny - 1), 2:(nz - (1 + corner_z[2]))) free_slip_x!(T)
    end
    if bc_y
        @parallel ((2 - corner_x[1]):(nx - 1), 2:(nz - (1 + corner_z[2]))) free_slip_y!(T)
    end
    if bc_z
        @parallel ((2 - corner_x[1]):(nx - 1), 2:(ny - (1 + corner_y[2]))) free_slip_z!(T)
    end
end
