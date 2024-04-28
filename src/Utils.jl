# MACROS
"""
    @idx(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro idx(args...)
    return quote
        _idx($(esc.(args)...))
    end
end

@inline _idx(args::NTuple{N,Int}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline _idx(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))

"""
    copy(B, A)

convenience macro to copy data from the array `A` into array `B`
"""
macro copy(B, A)
    return quote
        multi_copyto!($(esc(B)), $(esc(A)))
    end
end

multi_copyto!(B::AbstractArray, A::AbstractArray) = copyto!(B, A)

function detect_arsg_size(A::NTuple{N,AbstractArray{T,Dims}}) where {N,T,Dims}
    ntuple(Val(Dims)) do i
        Base.@_inline_meta
        s = ntuple(Val(N)) do j
            Base.@_inline_meta
            size(A[j], i)
        end
        maximum(s)
    end
end

@parallel_indices (I...) function multi_copy!(
    dst::NTuple{N,T}, src::NTuple{N,T}
) where {N,T}
    ntuple(Val(N)) do k
        Base.@_inline_meta
        if all(I .≤ size(dst[k]))
            @inbounds dst[k][I...] = src[k][I...]
        end
    end
    return nothing
end

Base.@propagate_inbounds @generated function unrolled_copy!(
    dst::NTuple{N,T}, src::NTuple{N,T}, I::Vararg{Int,NI}
) where {N,NI,T}
    quote
        Base.@_inline_meta
        Base.@nexprs $N n -> begin
            if all(tuple(I...) .≤ size(dst[n]))
                dst[n][I...] = src[n][I...]
            end
        end
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

@inline _tuple(V::JustRelax.Velocity{<:AbstractArray{T,2}}) where {T} = V.Vx, V.Vy
@inline _tuple(V::JustRelax.Velocity{<:AbstractArray{T,3}}) where {T} = V.Vx, V.Vy, V.Vz
@inline _tuple(A::JustRelax.SymmetricTensor{<:AbstractArray{T,2}}) where {T} =
    A.xx, A.yy, A.xy_c
@inline function _tuple(A::JustRelax.SymmetricTensor{<:AbstractArray{T,3}}) where {T}
    return A.xx, A.yy, A.zz, A.yz_c, A.xz_c, A.xy_c
end

"""
    @velocity(V)

Unpacks the velocity arrays `V` from the StokesArrays `A`.
"""
macro velocity(A)
    return quote
        unpack_velocity(($(esc(A))).V)
    end
end

@inline unpack_velocity(V::JustRelax.Velocity{<:AbstractArray{T,2}}) where {T} = V.Vx, V.Vy
@inline unpack_velocity(V::JustRelax.Velocity{<:AbstractArray{T,3}}) where {T} =
    V.Vx, V.Vy, V.Vz

"""
    @qT(V)

Unpacks the flux arrays `qT_i` from the ThermalArrays `A`.
"""
macro qT(A)
    return quote
        unpack_qT(($(esc(A))))
    end
end

@inline unpack_qT(A::JustRelax.ThermalArrays{<:AbstractArray{T,2}}) where {T} = A.qTx, A.qTy
@inline unpack_qT(A::JustRelax.ThermalArrays{<:AbstractArray{T,3}}) where {T} =
    A.qTx, A.qTy, A.qTz

"""
    @qT2(V)

Unpacks the flux arrays `qT2_i` from the ThermalArrays `A`.
"""
macro qT2(A)
    return quote
        unpack_qT2(($(esc(A))))
    end
end

@inline unpack_qT2(A::JustRelax.ThermalArrays{<:AbstractArray{T,2}}) where {T} =
    A.qTx2, A.qTy2
@inline function unpack_qT2(A::JustRelax.ThermalArrays{<:AbstractArray{T,3}}) where {T}
    return A.qTx2, A.qTy2, A.qTz2
end

"""
    @strain(A)

Unpacks the strain rate tensor `ε` from the StokesArrays `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro strain(A)
    return quote
        unpack_tensor_stag(($(esc(A))).ε)
    end
end

"""
    @strain_plastic(A)

Unpacks the plastic strain rate tensor `ε_pl` from the StokesArrays `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro strain_plastic(A)
    return quote
        unpack_tensor_stag(($(esc(A))).ε_pl)
    end
end

"""
    @stress(A)

Unpacks the deviatoric stress tensor `τ` from the StokesArrays `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro stress(A)
    return quote
        unpack_tensor_stag(($(esc(A))).τ)
    end
end

"""
    @tensor(A)

Unpacks the symmetric tensor `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro tensor(A)
    return quote
        unpack_tensor_stag(($(esc(A))))
    end
end

@inline function unpack_tensor_stag(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,2}}
) where {T}
    return A.xx, A.yy, A.xy
end
@inline function unpack_tensor_stag(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,3}}
) where {T}
    return A.xx, A.yy, A.zz, A.yz, A.xz, A.xy
end

"""
    @shear(A)

Unpacks the shear components of the symmetric tensor `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro shear(A)
    return quote
        unpack_shear_components_stag(($(esc(A))))
    end
end

@inline function unpack_shear_components_stag(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,2}}
) where {T}
    return A.xy
end
@inline function unpack_shear_components_stag(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,3}}
) where {T}
    return A.yz, A.xz, A.xy
end

"""
    @normal(A)

Unpacks the normal components of the symmetric tensor `A`, where its components are defined in the staggered grid.
Shear components are unpack following Voigt's notation.
"""
macro normal(A)
    return quote
        unpack_normal_components_stag(($(esc(A))))
    end
end

@generated function unpack_normal_components_stag(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,N}}
) where {T,N}
    syms = (:xx, :yy, :zz)
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> f_i = getfield(A, $syms[i])
        Base.@ncall $N tuple f
    end
end

"""
    @strain_center(A)

Unpacks the strain rate tensor `ε` from the StokesArrays `A`, where its components are defined in the center of the grid cells.
Shear components are unpack following Voigt's notation.
"""
macro strain_center(A)
    return quote
        unpack_tensor_center(($(esc(A))).ε)
    end
end

"""
    @stress_center(A)

Unpacks the deviatoric stress tensor `τ` from the StokesArrays `A`, where its components are defined in the center of the grid cells.
Shear components are unpack following Voigt's notation.
"""
macro stress_center(A)
    return quote
        unpack_tensor_center(($(esc(A))).τ)
    end
end

"""
    @tensor_center(A)

Unpacks the symmetric tensor `A`, where its components are defined in the center of the grid cells.
Shear components are unpack following Voigt's notation.
"""
macro tensor_center(A)
    return quote
        unpack_tensor_center(($(esc(A))))
    end
end

@inline function unpack_tensor_center(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,2}}
) where {T}
    return A.xx, A.yy, A.xy_c
end
@inline function unpack_tensor_center(
    A::JustRelax.SymmetricTensor{<:AbstractArray{T,3}}
) where {T}
    return A.xx, A.yy, A.zz, A.yz_c, A.xz_c, A.xy_c
end

"""
    @residuals(A)

Unpacks the momentum residuals from `A`.
"""
macro residuals(A)
    return quote
        unpack_residuals(($(esc(A))))
    end
end

@inline function unpack_residuals(
    A::JustRelax.Residual{<:AbstractArray{T,2}}
) where {T}
    return A.Rx, A.Ry
end
@inline function unpack_residuals(
    A::JustRelax.Residual{<:AbstractArray{T,3}}
) where {T}
    return A.Rx, A.Ry, A.Rz
end

## Memory allocators

macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end

function indices(::NTuple{3,T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    return i, j, k
end

function indices(::NTuple{2,T}) where {T}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return i, j
end

"""
    maxloc!(B, A; window)

Compute the maximum value of `A` in the `window = (width_x, width_y, width_z)` and store the result in `B`.
"""
function compute_maxloc!(B, A; window=(1, 1, 1))
    ni = size(A)

    @parallel_indices (I...) function _maxloc!(B, A, window)
        B[I...] = _maxloc_window_clamped(A, I..., window...)
        return nothing
    end

    @parallel (@idx ni) _maxloc!(B, A, window)
    return nothing
end

@inline function _maxloc_window_clamped(A, I, J, width_x, width_y)
    nx, ny = size(A)
    I_range = (I - width_x):(I + width_x)
    J_range = (J - width_y):(J + width_y)
    x = -Inf

    for j in J_range
        jj = clamp(j, 1, ny) # handle boundary cells
        for i in I_range
            ii = clamp(i, 1, nx) # handle boundary cells
            Aij = A[ii, jj]
            if Aij > x
                x = Aij
            end
        end
    end
    return x
end

@inline function _maxloc_window_clamped(A, I, J, K, width_x, width_y, width_z)
    nx, ny, nz = size(A)
    I_range = (I - width_x):(I + width_x)
    J_range = (J - width_y):(J + width_y)
    K_range = (K - width_z):(K + width_z)
    x = -Inf

    for k in K_range
        kk = clamp(k, 1, nz) # handle boundary cells
        for j in J_range
            jj = clamp(j, 1, ny) # handle boundary cells
            for i in I_range
                ii = clamp(i, 1, nx) # handle boundary cells
                Aijk = A[ii, jj, kk]
                if Aijk > x
                    x = Aijk
                end
            end
        end
    end
    return x
end

# unpacks fields of the struct x into a tuple
@generated function unpack(x::T) where {T}
    return quote
        Base.@_inline_meta
        tuple(_unpack(x, fieldnames($T))...)
    end
end
_unpack(a, fields) = (getfield(a, fi) for fi in fields)

macro unpack(x)
    return quote
        unpack($(esc(x)))
    end
end

"""
    compute_dt(S::JustRelax.StokesArrays, di)

Compute the time step `dt` for the velocity field `S.V` for a regular grid with grid spacing `di`.
"""
@inline compute_dt(S::JustRelax.StokesArrays, di) = compute_dt(@velocity(S), di, Inf)

"""
    compute_dt(S::JustRelax.StokesArrays, di, dt_diff)

Compute the time step `dt` for the velocity field `S.V` and the diffusive maximum time step
`dt_diff` for a regular gridwith grid spacing `di`.
"""
@inline compute_dt(S::JustRelax.StokesArrays, di, dt_diff) =
    compute_dt(@velocity(S), di, dt_diff)

@inline function compute_dt(V::NTuple, di, dt_diff)
    n = inv(length(V) + 0.1)
    dt_adv = mapreduce(x -> x[1] * inv(maximum_mpi(abs.(x[2]))), min, zip(di, V)) * n
    return min(dt_diff, dt_adv)
end
"""
    compute_dt(S::JustRelax.StokesArrays, di, igg)

Compute the time step `dt` for the velocity field `S.V` for a regular gridwith grid spacing `di`.
The implicit global grid variable `I` implies that the time step is calculated globally and not
separately on each block.
"""
@inline compute_dt(S::JustRelax.StokesArrays, di, I::IGG) =
    compute_dt(@velocity(S), di, Inf, I::IGG)

"""
    compute_dt(S::JustRelax.StokesArrays, di, dt_diff)

Compute the time step `dt` for the velocity field `S.V` and the diffusive maximum time step
`dt_diff` for a regular gridwith grid spacing `di`. The implicit global grid variable `I`
implies that the time step is calculated globally and not separately on each block.
"""
@inline function compute_dt(S::JustRelax.StokesArrays, di, dt_diff, I::IGG)
    return compute_dt(@velocity(S), di, dt_diff, I::IGG)
end

@inline function compute_dt(V::NTuple, di, dt_diff, I::IGG)
    n = inv(length(V) + 0.1)
    dt_adv = mapreduce(x -> x[1] * inv(maximum_mpi(abs.(x[2]))), max, zip(di, V)) * n
    return min(dt_diff, dt_adv)
end

@inline tupleize(v) = (v,)
@inline tupleize(v::Tuple) = v

"""
    allzero(x::Vararg{T,N}) where {T,N}

Check if all elements in `x` are zero.

# Arguments
- `x::Vararg{T,N}`: The input array.

# Returns
- `Bool`: `true` if all elements in `x` are zero, `false` otherwise.
"""
@inline allzero(x::Vararg{T,N}) where {T,N} = all(x -> x == 0, x)

"""
    take(fldr::String)

Create folder `fldr` if it does not exist.
"""
take(fldr::String) = !isdir(fldr) && mkpath(fldr)

"""
    continuation_log(x_new, x_old, ν)

Do a continuation step `exp((1-ν)*log(x_old) + ν*log(x_new))` with damping parameter `ν`
"""
@inline function continuation_log(x_new::T, x_old::T, ν) where {T}
    x_cont = exp((1 - ν) * log(x_old) + ν * log(x_new))
    return isnan(x_cont) ? 0.0 : x_cont
end
@inline continuation_linear(x_new, x_old, ν) = muladd((1 - ν), x_old, ν * x_new) # (1 - ν) * x_old + ν * x_new

# Others

"""
    assign!(B::AbstractArray{T,N}, A::AbstractArray{T,N}) where {T,N}

Assigns the values of array `A` to array `B` in parallel.

# Arguments
- `B::AbstractArray{T,N}`: The destination array.
- `A::AbstractArray{T,N}`: The source array.
"""
@parallel function assign!(B::AbstractArray{T,N}, A::AbstractArray{T,N}) where {T,N}
    @all(B) = @all(A)
    return nothing
end

# MPI reductions

function mean_mpi(A)
    mean_l = _mean(A)
    return MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD) / MPI.Comm_size(MPI.COMM_WORLD)
end

function norm_mpi(A)
    sum2_l = _sum(A .^ 2)
    return sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD))
end

function minimum_mpi(A)
    min_l = _minimum(A)
    return MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD)
end

function maximum_mpi(A)
    max_l = _maximum(A)
    return MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD)
end

for (f1, f2) in zip(
    (:_mean, :_norm, :_minimum, :_maximum, :_sum), (:mean, :norm, :minimum, :maximum, :sum)
)
    @eval begin
        $f1(A::AbstractArray) = $f2(Array(A))
        $f1(A) = $f2(A)
    end
end
