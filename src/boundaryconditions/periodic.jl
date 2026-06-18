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
