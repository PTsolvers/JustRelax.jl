# MACROS

export @allocate, @fill

# Memory allocators
macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end

macro fill(A, ni...)
    return esc(:(PTArray(fill(eltype(PTArray)($A), $(ni...)))))
end

# MPI REDUCTIONS 

export mean_mpi, norm_mpi, minimum_mpi, maximum_mpi

function mean_mpi(A)
    mean_l = mean(A)
    return MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD) / MPI.Comm_size(MPI.COMM_WORLD)
end

function norm_mpi(A)
    sum2_l = sum(A .^ 2)
    return sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD))
end

function minimum_mpi(A)
    min_l = maximum(A)
    return MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD)
end

function maximum_mpi(A)
    max_l = minimum(A)
    return MPI.Allreduce(max_l, MPI.MIN, MPI.COMM_WORLD)
end
