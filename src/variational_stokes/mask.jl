function RockRatio(::Type{CPUBackend}, ni::NTuple{N,Integer}) where {N}
    return RockRatio(ni...)
end

function RockRatio(nx, ny)
    ni = nx, ny
    center = @zeros(ni...)
    vertex = @zeros(ni .+ 1...)
    Vx = @zeros(nx + 1, ny) # no ghost nodes!
    Vy = @zeros(nx, ny + 1) # no ghost nodes!
    dummy = @zeros(1, 1) # because it cant be a Union{T, Nothing} type on the GPU....
    return JustRelax.RockRatio(center, vertex, Vx, Vy, dummy, dummy, dummy, dummy)
end

function RockRatio(nx, ny, nz)
    ni = nx, ny, nz
    center = @zeros(ni...)
    vertex = @zeros(ni .+ 1...)
    Vx = @zeros(nx + 1, ny, nz) # no ghost nodes!
    Vy = @zeros(nx, ny + 1, nz) # no ghost nodes!
    Vz = @zeros(nx, ny, nz + 1) # no ghost nodes!
    yz = @zeros(nx, ny + 1, nz + 1)
    xz = @zeros(nx + 1, ny, nz + 1)
    xy = @zeros(nx + 1, ny + 1, nz)

    return JustRelax.RockRatio(center, vertex, Vx, Vy, Vz, yz, xz, xy)
end

@inline size_c(x::JustRelax.AbstractMask) = size(x.center)
@inline size_v(x::JustRelax.AbstractMask) = size(x.vertex)
@inline size_vx(x::JustRelax.AbstractMask) = size(x.Vx)
@inline size_vy(x::JustRelax.AbstractMask) = size(x.Vy)
@inline size_vz(x::JustRelax.AbstractMask) = size(x.Vz)
@inline size_yz(x::JustRelax.AbstractMask) = size(x.yz)
@inline size_xz(x::JustRelax.AbstractMask) = size(x.xz)
@inline size_xy(x::JustRelax.AbstractMask) = size(x.xy)

"""
    update_rock_ratio!(ϕ::JustRelax.RockRatio, phase_ratios, air_phase)

Update the rock ratio `ϕ` based on the provided `phase_ratios` and `air_phase`.

# Arguments
- `ϕ::JustRelax.RockRatio`: The rock ratio object to be updated.
- `phase_ratios`: The ratios of different phases present.
- `air_phase`: The phase representing air.
"""
function update_rock_ratio!(ϕ::JustRelax.RockRatio{T,2}, phase_ratios, air_phase) where {T}
    nvi = size_v(ϕ)
    @parallel (@idx nvi) update_rock_ratio_cv!(
        ϕ, phase_ratios.center, phase_ratios.vertex, air_phase
    )

    dst = ϕ.Vx, ϕ.Vy
    src = phase_ratios.Vx, phase_ratios.Vy

    for (dstᵢ, srcᵢ) in zip(dst, src)
        @parallel (@idx size(dstᵢ)) _update_rock_ratio!(dstᵢ, srcᵢ, air_phase)
    end

    return nothing
end

function update_rock_ratio!(ϕ::JustRelax.RockRatio{T,3}, phase_ratios, air_phase) where {T}
    nvi = size_v(ϕ)
    @parallel (@idx nvi) update_rock_ratio_cv!(
        ϕ, phase_ratios.center, phase_ratios.vertex, air_phase
    )

    dst = ϕ.Vx, ϕ.Vy, ϕ.Vz, ϕ.xy, ϕ.yz, ϕ.xz
    src = phase_ratios.Vx,
    phase_ratios.Vy, phase_ratios.Vz, phase_ratios.xy, phase_ratios.yz,
    phase_ratios.xz

    for (dstᵢ, srcᵢ) in zip(dst, src)
        @parallel (@idx size(dstᵢ)) _update_rock_ratio!(dstᵢ, srcᵢ, air_phase)
    end

    return nothing
end

@inline function compute_rock_ratio(
    phase_ratio::CellArray, air_phase, I::Vararg{Integer,N}
) where {N}
    1 ≤ air_phase ≤ numphases(phase_ratio) || return 1e0
    x = 1 - @index phase_ratio[air_phase, I...]
    x *= x > 1e-5
    return x
end

@inline function compute_air_ratio(
    phase_ratio::CellArray, air_phase, I::Vararg{Integer,N}
) where {N}
    1 ≤ air_phase ≤ numphases(phase_ratio) || return 1e0
    return @index phase_ratio[air_phase, I...]
end

@parallel_indices (I...) function update_rock_ratio_cv!(
    ϕ, ratio_center, ratio_vertex, air_phase
)
    if all(I .≤ size(ratio_center))
        ϕ.center[I...] = compute_rock_ratio(ratio_center, air_phase, I...)
    end
    ϕ.vertex[I...] = compute_rock_ratio(ratio_vertex, air_phase, I...)
    return nothing
end

@parallel_indices (I...) function _update_rock_ratio!(ϕ, ratio, air_phase)
    # ϕ[I...] = Float64(Float16(compute_rock_ratio(ratio, air_phase, I...)))
    ϕ[I...] = clamp(compute_rock_ratio(ratio, air_phase, I...), 0, 1)
    return nothing
end

"""
    isvalid_c(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.center[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j)
    vx = isvalid(ϕ.Vx, i, j) * isvalid(ϕ.Vx[i + 1, j])
    vy = isvalid(ϕ.Vy, i, j) * isvalid(ϕ.Vy[i, j + 1])
    v = vx * vy
    return v * isvalid(ϕ.center, i, j)
end

Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j, k)
    vx = isvalid(ϕ.Vx, i, j, k) * isvalid(ϕ.Vx, i + 1, j, k)
    vy = isvalid(ϕ.Vy, i, j, k) * isvalid(ϕ.Vy, i, j + 1, k)
    vz = isvalid(ϕ.Vz, i, j, k) * isvalid(ϕ.Vz, i, j, k + 1)
    v = vx * vy * vz
    return v * isvalid(ϕ.center, i, j, k)
end

"""
    isvalid_v(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.vertex[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_v(ϕ::JustRelax.RockRatio, i, j)
    nx, ny = size(ϕ.Vx)
    j_bot = max(j - 1, 1)
    j0 = min(j, ny)
    vx = isvalid(ϕ.Vx, i, j0) * isvalid(ϕ.Vx, i, j_bot)

    nx, ny = size(ϕ.Vy)
    i_left = max(i - 1, 1)
    i0 = min(i, nx)
    vy = isvalid(ϕ.Vy, i0, j) * isvalid(ϕ.Vy, i_left, j)
    v = vx * vy
    return v * isvalid(ϕ.vertex, i, j)
end

"""
    isvalid_vx(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.Vx[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_vx(
    ϕ::JustRelax.RockRatio, I::Vararg{Integer,N}
) where {N}
    return isvalid(ϕ.Vx, I...)
end

# Base.@propagate_inbounds @inline function isvalid_vx(ϕ::JustRelax.RockRatio, I::Vararg{Integer,N}) where {N}
#     # c = (ϕ.center[i, j] > 0) * (ϕ.center[i - 1, j] > 0)
#     # v = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i, j + 1] > 0)
#     # cv = c * v
#     # return cv * (ϕ.Vx[i, j] > 0)
#     return (ϕ.Vx[I...] > 0)
# end

"""
    isvalid_vy(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.Vy[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
# Base.@propagate_inbounds @inline function isvalid_vy(ϕ::JustRelax.RockRatio, i, j)
#     # c = (ϕ.center[i, j] > 0) * (ϕ.center[i, j - 1] > 0)
#     # v = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i + 1, j] > 0)
#     # cv = c * v
#     # return cv * (ϕ.Vy[i, j] > 0)
#     return (ϕ.Vy[i, j] > 0)
# end
Base.@propagate_inbounds @inline function isvalid_vy(
    ϕ::JustRelax.RockRatio, I::Vararg{Integer,N}
) where {N}
    return isvalid(ϕ.Vy, I...)
end

"""
    isvalid_vz(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.Vz[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_vz(
    ϕ::JustRelax.RockRatio, I::Vararg{Integer,N}
) where {N}
    return isvalid(ϕ.Vz, I...)
end

Base.@propagate_inbounds @inline function isvalid_velocity(ϕ::JustRelax.RockRatio, i, j)
    return isvalid(ϕ.Vx, i, j) * isvalid(ϕ.Vy, i, j)
end

Base.@propagate_inbounds @inline function isvalid_velocity(ϕ::JustRelax.RockRatio, i, j, k)
    return isvalid(ϕ.Vx, i, j, k) * isvalid(ϕ.Vy, i, j, k) * isvalid(ϕ.Vz, i, j, k)
end

Base.@propagate_inbounds @inline function isvalid_v(ϕ::JustRelax.RockRatio, i, j, k)
    # yz
    nx, ny, nz = size(ϕ.yz)
    i_left = max(i - 1, 1)
    i_right = min(i, nx)
    yz = isvalid(ϕ.yz, i_left, j, k) * isvalid(ϕ.yz, i_right, j, k)

    # xz
    nx, ny, nz = size(ϕ.xz)
    j_front = max(j - 1, 1)
    j_back = min(j, ny)
    xz = isvalid(ϕ.xz, i, j_front, k) * isvalid(ϕ.xz, i, j_back, k)

    # xy
    nx, ny, nz = size(ϕ.xy)
    k_top = max(k - 1, 1)
    k_bot = min(k, nz)
    xy = isvalid(ϕ.xy, i, j, k_top) * isvalid(ϕ.xy, i, j, k_back)

    # V
    v = yz * xz * xy

    return v * isvalid(ϕ.vertex, i, j, k)
end

Base.@propagate_inbounds @inline function isvalid_xz(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i, j + 1, k)

    # check vz
    nx, ny, nz = size(ϕ.vz)
    i_left = max(i - 1, 1)
    i_right = min(i, nx)
    vz = isvalid(ϕ.vz, i_left, j, k) * isvalid(ϕ.vz, i_right, j, k)

    # check vx
    nx, ny, nz = size(ϕ.vx)
    k_top = max(k - 1, 1)
    k_bot = min(k, nz)
    vx = isvalid(ϕ.vx, i, j, k_top) * isvalid(ϕ.vx, i, j, k_back)

    return v * vx * vz * isvalid(ϕ.vertex, i, j, k)
end

Base.@propagate_inbounds @inline function isvalid_xy(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i, j, k + 1)

    # check vx
    nx, ny, nz = size(ϕ.vx)
    j_front = max(j - 1, 1)
    j_back = min(j, ny)
    vx = isvalid(ϕ.vx, i, j_front, k) * isvalid(ϕ.vx, i, j_back, k)

    # check vy
    nx, ny, nz = size(ϕ.vy)
    i_left = max(i - 1, 1)
    i_right = min(i, nx)
    vy = isvalid(ϕ.vy, i_left, j, k) * isvalid(ϕ.vy, i_right, j, k)

    return v * vy * vz * isvalid(ϕ.vertex, i, j, k)
end

Base.@propagate_inbounds @inline function isvalid_yz(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i + 1, j, k)

    # check vz
    nx, ny, nz = size(ϕ.vz)
    j_front = max(j - 1, 1)
    j_back = min(j, ny)
    vz = isvalid(ϕ.vz, i, j_front, k) * isvalid(ϕ.vz, i, j_back, k)

    # check vy
    nx, ny, nz = size(ϕ.vy)
    k_top = max(k - 1, 1)
    k_bot = min(k, nz)
    vy = isvalid(ϕ.vy, i, j, k_top) * isvalid(ϕ.vy, i, j, k_back)

    return v * vy * vz * isvalid(ϕ.vertex, i, j, k)
end

Base.@propagate_inbounds @inline isvalid(ϕ, I::Vararg{Integer,N}) where {N} = ϕ[I...] > 0

######

# """
#     isvalid_c(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.center[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j)
#     return isvalid(ϕ.center, i, j)
# end

# """
#     isvalid_v(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.vertex[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_v(ϕ::JustRelax.RockRatio, i, j)
#     return isvalid(ϕ.vertex, i, j)
# end

# """
#     isvalid_vx(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.Vx[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_vx(ϕ::JustRelax.RockRatio, i, j)
#     c = isvalid(ϕ.center, i, j) || isvalid(ϕ.center, i - 1, j)
#     v = isvalid(ϕ.vertex, i, j) || isvalid(ϕ.vertex, i, j + 1)
#     cv = c || v
#     return cv || isvalid(ϕ.Vx, i, j)
# end

# """
#     isvalid_vy(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.Vy[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_vy(ϕ::JustRelax.RockRatio, i, j)
#     c = isvalid(ϕ.center, i, j) || isvalid(ϕ.center, i, j - 1)
#     v = isvalid(ϕ.vertex, i, j) || isvalid(ϕ.vertex, i + 1, j)
#     cv = c || v
#     return cv || isvalid(ϕ.Vy, i, j)
# end
