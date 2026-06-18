@parallel_indices (i) function dirichlet_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 2}}
    @inbounds begin
        if i ≤ size(T, 1)
            T[i, 1] = bc.bot === false ? T[i, 1] : 2 * bc.bot - T[i, 2]
            T[i, end] = bc.top === false ? T[i, end] : 2 * bc.top - T[i, end - 1]
        end
        if i ≤ size(T, 2)
            T[1, i] = bc.left === false ? T[1, i] : 2 * bc.left - T[2, i]
            T[end, i] = bc.right === false ? T[end, i] : 2 * bc.right - T[end - 1, i]
        end
    end
    return nothing
end

@parallel_indices (i, j) function dirichlet_boundary!(T::_T, bc) where {_T <: AbstractArray{<:Any, 3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            T[i, j, 1] = bc.bot === false ? T[i, j, 1] : 2 * bc.bot - T[i, j, 2]
            T[i, j, end] = bc.top === false ? T[i, j, end] : 2 * bc.top - T[i, j, end - 1]
        end
        if i ≤ ny && j ≤ nz
            T[1, i, j] = bc.left === false ? T[1, i, j] : 2 * bc.left - T[2, i, j]
            T[end, i, j] = bc.right === false ? T[end, i, j] : 2 * bc.right - T[end - 1, i, j]
        end
        if i ≤ nx && j ≤ nz
            T[i, 1, j] = bc.front === false ? T[i, 1, j] : 2 * bc.front - T[i, 2, j]
            T[i, end, j] = bc.back === false ? T[i, end, j] : 2 * bc.back - T[i, end - 1, j]
        end
    end
    return nothing
end
