@parallel_indices (i) function periodic_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, end - 1])
            bc.top && (T[i, end] = T[i, 2])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[end - 1, i])
            bc.right && (T[end, i] = T[2, i])
        end
    end
    return nothing
end

@parallel_indices (i) function periodic_boundary!(Vx, Vy, bc)
    @inbounds begin
        if i ≤ size(Vx, 2)
            # Normal component: paired faces represent the same physical plane.
            bc.left && (Vx[1, i] = Vx[end, i])
        end
        if i ≤ size(Vy, 2)
            bc.left && (Vy[1, i] = Vy[end - 1, i])
            bc.right && (Vy[end, i] = Vy[2, i])
        end
        if i ≤ size(Vx, 1)
            bc.bot && (Vx[i, 1] = Vx[i, end - 1])
            bc.top && (Vx[i, end] = Vx[i, 2])
        end
        if i ≤ size(Vy, 1)
            # Normal component: paired faces represent the same physical plane.
            bc.bot && (Vy[i, 1] = Vy[i, end])
        end
    end
    return nothing
end

@parallel_indices (i, j) function periodic_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, end - 1])
            bc.top && (T[i, j, end] = T[i, j, 2])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[end - 1, i, j])
            bc.right && (T[end, i, j] = T[2, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, end - 1, j])
            bc.back && (T[i, end, j] = T[i, 2, j])
        end
    end
    return nothing
end

@parallel_indices (i, j) function periodic_boundary!(Vx, Vy, Vz, bc)
    @inbounds begin
        if i ≤ size(Vx, 2) && j ≤ size(Vx, 3)
            # Normal component: paired faces represent the same physical plane.
            bc.left && (Vx[1, i, j] = Vx[end, i, j])
        end
        if i ≤ size(Vy, 2) && j ≤ size(Vy, 3)
            bc.left && (Vy[1, i, j] = Vy[end - 1, i, j])
            bc.right && (Vy[end, i, j] = Vy[2, i, j])
        end
        if i ≤ size(Vz, 2) && j ≤ size(Vz, 3)
            bc.left && (Vz[1, i, j] = Vz[end - 1, i, j])
            bc.right && (Vz[end, i, j] = Vz[2, i, j])
        end

        if i ≤ size(Vx, 1) && j ≤ size(Vx, 3)
            bc.front && (Vx[i, 1, j] = Vx[i, end - 1, j])
            bc.back && (Vx[i, end, j] = Vx[i, 2, j])
        end
        if i ≤ size(Vy, 1) && j ≤ size(Vy, 3)
            # Normal component: paired faces represent the same physical plane.
            bc.front && (Vy[i, 1, j] = Vy[i, end, j])
        end
        if i ≤ size(Vz, 1) && j ≤ size(Vz, 3)
            bc.front && (Vz[i, 1, j] = Vz[i, end - 1, j])
            bc.back && (Vz[i, end, j] = Vz[i, 2, j])
        end

        if i ≤ size(Vx, 1) && j ≤ size(Vx, 2)
            bc.bot && (Vx[i, j, 1] = Vx[i, j, end - 1])
            bc.top && (Vx[i, j, end] = Vx[i, j, 2])
        end
        if i ≤ size(Vy, 1) && j ≤ size(Vy, 2)
            bc.bot && (Vy[i, j, 1] = Vy[i, j, end - 1])
            bc.top && (Vy[i, j, end] = Vy[i, j, 2])
        end
        if i ≤ size(Vz, 1) && j ≤ size(Vz, 2)
            # Normal component: paired faces represent the same physical plane.
            bc.bot && (Vz[i, j, 1] = Vz[i, j, end])
        end
    end
    return nothing
end
