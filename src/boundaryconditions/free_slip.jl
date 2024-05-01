@parallel_indices (i) function free_slip!(Ax, Ay, bc)
    @inbounds begin
        if i ≤ size(Ax, 1)
            bc.bot && (Ax[i, 1] = Ax[i, 2])
            bc.top && (Ax[i, end] = Ax[i, end - 1])
        end
        if i ≤ size(Ay, 2)
            bc.left && (Ay[1, i] = Ay[2, i])
            bc.right && (Ay[end, i] = Ay[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(Ax, Ay, Az, bc)
    @inbounds begin
        # free slip in the front and back XZ planes
        if bc.front
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, 1, j] = Ax[i, 2, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, 1, j] = Az[i, 2, j]
            end
        end
        if bc.back
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, end, j] = Ax[i, end - 1, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, end, j] = Az[i, end - 1, j]
            end
        end
        # free slip in the front and back XY planes
        if bc.top
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, 1] = Ax[i, j, 2]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, 1] = Ay[i, j, 2]
            end
        end
        if bc.bot
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, end] = Ax[i, j, end - 1]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, end] = Ay[i, j, end - 1]
            end
        end
        # free slip in the front and back YZ planes
        if bc.left
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[1, i, j] = Ay[2, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[1, i, j] = Az[2, i, j]
            end
        end
        if bc.right
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[end, i, j] = Ay[end - 1, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[end, i, j] = Az[end - 1, i, j]
            end
        end
    end
    return nothing
end

@parallel_indices (i) function free_slip!(T::_T, bc) where {_T<:AbstractArray{<:Any,2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, 2])
            bc.top && (T[i, end] = T[i, end - 1])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[2, i])
            bc.right && (T[end, i] = T[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(T::_T, bc) where {_T<:AbstractArray{<:Any,3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, 2])
            bc.top && (T[i, j, end] = T[i, j, end - 1])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[2, i, j])
            bc.right && (T[end, i, j] = T[end - 1, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, 2, j])
            bc.back && (T[i, end, j] = T[i, end - 1, j])
        end
    end
    return nothing
end



@parallel_indices (j) function free_slip_x!(A::AbstractArray{T,2}) where {T}
    A[1, j] = A[2, j]
    A[end, j] = A[end - 1, j]
    return nothing
end

@parallel_indices (i) function free_slip_y!(A::AbstractArray{T,2}) where {T}
    A[i, 1] = A[i, 2]
    A[i, end] = A[i, end - 1]
    return nothing
end

@inbounds @parallel_indices (i) function _apply_free_slip!(Ax, Ay, freeslip_x, freeslip_y)
    if freeslip_x && i ≤ size(Ax, 1)
        Ax[i, 1] = Ax[i, 2]
        Ax[i, end] = Ax[i, end - 1]
    end
    if freeslip_y && i ≤ size(Ay, 2)
        Ay[1, i] = Ay[2, i]
        Ay[end, i] = Ay[end - 1, i]
    end
    return nothing
end

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy) where {T}
    freeslip_x, freeslip_y = freeslip
    n = max(size(Vx, 1), size(Vy, 2))
    # free slip boundary conditions
    @parallel (1:n) _apply_free_slip!(Vx, Vy, freeslip_x, freeslip_y)

    return nothing
end

# 3D KERNELS

@parallel_indices (j, k) function free_slip_x!(A::AbstractArray{T,3}) where {T}
    A[1, j, k] = A[2, j, k]
    A[end, j, k] = A[end - 1, j, k]
    return nothing
end

@parallel_indices (i, k) function free_slip_y!(A::AbstractArray{T,3}) where {T}
    A[i, 1, k] = A[i, 2, k]
    A[i, end, k] = A[i, end - 1, k]
    return nothing
end

@parallel_indices (i, j) function free_slip_z!(A::AbstractArray{T,3}) where {T}
    A[i, j, 1] = A[i, j, 2]
    A[i, j, end] = A[i, j, end - 1]
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