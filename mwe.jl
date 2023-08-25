using ParallelStencil
using CUDA
@init_parallel_stencil(CUDA, Float64, 2)

function indices(::NTuple{3, T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    return i, j, k
end

function indices(::NTuple{2, T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return i, j
end

function _maxloc(A, i, j)
    x = -Inf
    for ii in i-1:i+1, jj in j-1:j+1
        Aij = A[ii, jj]
        if Aij > x
            x = Aij
        end
    end
    x
end

function _maxloc(A, i, j, k)
    x = -Inf
    for ii in i-1:i+1, jj in j-1:j+1, kk in k-1:k+1
        Aij = A[ii, jj, kk]
        if Aij > x
            x = Aij
        end
    end
    x
end

function _foo1!(B, A)

    I = indices(size(B))
    
    # if all(1 .< I .< size(A)) 
    #     B[I...] = _maxloc(A, I...)
    # end

    return nothing
end

function foo1!(B::CuArray{T1, N, T2}, A::CuArray{T1, N, T2}) where {T1, T2, N}
    ni = size(B)
    nthreads = ntuple(i->16, Val(N))
    nblocks = ntuple(i->ceil(Int, ni[i] / nthreads[i]), Val(N))
    @sync @cuda threads=nthreads blocks=nblocks _foo1!(B, A)
    return nothing
end


function foo2!(B::Array{T,N}, A::Array{T,N}) where {T,N}
    ni = size(A)
    idx = ntuple(i->2:ni[i]-1, Val(length(ni)))

    @parallel_indices (i, j) function _foo2!(B::AbstractArray{T, 2}, A) where T
        # B[i, j] = _maxloc(A, i, j)
        return nothing
    end


    @parallel_indices (i, j) function _foo3!(B::AbstractArray{T, 2}, A) where T
        # B[i, j] = _maxloc(A, i, j)
        return nothing
    end
    @parallel_indices (i, j, k) function _foo2!(B::AbstractArray{T, 3}, A) where T
        # B[i, j, k] = _maxloc(A, i, j, k)
        return nothing
    end

    @parallel (idx) _foo3!(B, A)
end


N = 700
A1 = @rand(N, N);
A2 = deepcopy(A1);
B1 = @zeros(N, N);
B2 = @zeros(N, N);

@btime foo1!($B1, $A1)
@btime foo2!($B2, $A2)

@assert B1 == B2


@parallel_indices (i, j) function _potato!(A) 
        # B[i, j] = _maxloc(A, i, j)
        return nothing
    end
    

@btime @parallel (1:$N, 1:$N) _potato!(A1)
