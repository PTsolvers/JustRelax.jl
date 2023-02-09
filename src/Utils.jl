@inline compute_dt(S::StokesArrays, di) = compute_dt(S.V, di, Inf)
@inline compute_dt(S::StokesArrays, di, dt_diff) = compute_dt(S.V, di, dt_diff)

@inline function compute_dt(V::Velocity, di::NTuple{2,T}, dt_diff) where {T}
    return compute_dt(V.Vx, V.Vy, di[1], di[2], dt_diff)
end

@inline function compute_dt(V::Velocity, di::NTuple{3,T}, dt_diff) where {T}
    return compute_dt(V.Vx, V.Vy, V.Vz, di[1], di[2], di[3], dt_diff)
end

@inline function compute_dt(Vx, Vy, dx, dy, dt_diff)
    dt_adv = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy))) / 2.1
    return min(dt_diff, dt_adv)
end

@inline function compute_dt(Vx, Vy, Vz, dx, dy, dz, dt_diff)
    dt_adv =
        min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 3.1
    return min(dt_diff, dt_adv)
end

@inline tupleize(v::MaterialParams) = (v,)
@inline tupleize(v::Tuple) = v

# MACROS

"""
    copy(B, A)

Convinience macro to copy data from the array `A` into array `B`
"""
macro copy(B, A)
    return quote
        copyto!($(esc(B)), $(esc(A)))
    end
end

"""
    @add(I, args...)

Add `I` to the scalars in `args`    
"""
macro add(I, args...)
    quote
        Base.@_inline_meta
        v = (; $(esc.(args)...))
        values(v) .+ $(esc(I))
    end
end

macro tuple(A)
    return quote
        _tuple($(esc(A)))
    end
end

_tuple(V::Velocity{<:AbstractArray{T,2}}) where {T} = V.Vx, V.Vy
_tuple(V::Velocity{<:AbstractArray{T,3}}) where {T} = V.Vx, V.Vy, V.Vz
_tuple(A::SymmetricTensor{<:AbstractArray{T,2}}) where {T} = A.xx, A.yy, A.xy_c
function _tuple(A::SymmetricTensor{<:AbstractArray{T,3}}) where {T}
    return A.xx, A.yy, A.zz, A.yz_c, A.xz_c, A.xy_c
end

"""
    @idx(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro idx(args...)
    return quote
        _idx(tuple($(esc.(args)...))...)
    end
end

@inline Base.@pure _idx(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline Base.@pure _idx(args::NTuple{N,Int}) where {N} = ntuple(i -> 1:args[i], Val(N))

## Memory allocators

macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end

# Others

@parallel function assign!(B::AbstractArray{T,N}, A::AbstractArray{T,N}) where {T,N}
    @all(B) = @all(A)
    return nothing
end

# MPI reductions 

function mean_mpi(A)
    mean_l = mean(A)
    return MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD) / MPI.Comm_size(MPI.COMM_WORLD)
end

function norm_mpi(A)
    sum2_l = sum(A .^ 2)
    return sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD))
end

function minimum_mpi(A)
    min_l = minimum(A)
    return MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD)
end

function maximum_mpi(A)
    max_l = maximum(A)
    return MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD)
end
