function RockRatio(::Type{CPUBackend}, ni::NTuple{N,Integer}) where {N}
    return RockRatio(ni...)
end

function RockRatio(nx, ny)
    ni = nx, ny
    center = @zeros(ni...)
    vertex = @zeros(ni .+ 1...)
    Vx = @zeros(nx+1, ny) # no ghost nodes!
    Vy = @zeros(nx, ny+1) # no ghost nodes!
    dummy = @zeros(1, 1) # because it cant be a Union{T, Nothing} type on the GPU....
    return JustRelax.RockRatio(center, vertex, Vx, Vy, dummy, dummy, dummy, dummy)
end

function RockRatio(nx, ny, nz)
    ni = nx, ny, nz
    center = @zeros(ni...)
    vertex = @zeros(ni .+ 1...)
    Vx = @zeros(nx+1, ny, nz) # no ghost nodes!
    Vy = @zeros(nx, ny+1, nz) # no ghost nodes!
    Vz = @zeros(nx, ny, nz+1) # no ghost nodes!
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
function update_rock_ratio!(ϕ::JustRelax.RockRatio, phase_ratios, ratio_vel::NTuple{N}, air_phase) where N
    nvi = size_v(ϕ)
    @parallel (@idx nvi) update_rock_ratio_cv!(
        ϕ, phase_ratios.center, phase_ratios.vertex, air_phase
    )
    # @parallel (@idx nvi) update_rock_ratio_vel!(ϕ)

    @parallel (@idx size(ϕ.Vx)) update_rock_ratio_vel!(ϕ.Vx, ratio_vel[1], air_phase)
    @parallel (@idx size(ϕ.Vy)) update_rock_ratio_vel!(ϕ.Vy, ratio_vel[2], air_phase)
    if N === 3
        @parallel (@idx size(ϕ.Vz)) update_rock_ratio_vel!(ϕ.Vz, ratio_vel[3], air_phase)
    end

    return nothing
end

@inline compute_rock_ratio(
    phase_ratio::CellArray, air_phase, I::Vararg{Integer,N}
) where {N} = 1 - @index phase_ratio[air_phase, I...]

@inline compute_air_ratio(phase_ratio::CellArray, air_phase, I::Vararg{Integer,N}) where {N} = @index phase_ratio[
    air_phase, I...
]

@parallel_indices (I...) function update_rock_ratio_cv!(
    ϕ, ratio_center, ratio_vertex, air_phase
)
    if all(I .≤ size(ratio_center))
        ϕ.center[I...] = Float64(Float16(compute_rock_ratio(ratio_center, air_phase, I...)))
    end
    ϕ.vertex[I...] = Float64(Float16(compute_rock_ratio(ratio_vertex, air_phase, I...)))
    return nothing
end

@parallel_indices (I...) function update_rock_ratio_vel!(ϕ, ratio, air_phase)
    ϕ[I...] = Float64(Float16(compute_rock_ratio(ratio, air_phase, I...)))
    return nothing
end



# """
#     isvalid_c(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.center[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j)
#     vx = (ϕ.Vx[i, j] > 0) * (ϕ.Vx[i + 1, j] > 0)
#     vy = (ϕ.Vy[i, j] > 0) * (ϕ.Vy[i, j + 1] > 0)
#     v = vx * vy
#     return v * (ϕ.center[i, j] > 0)
# end

# """
#     isvalid_v(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.vertex[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_v(ϕ::JustRelax.RockRatio, i, j)
#     nx, ny = size(ϕ.Vx)
#     j_bot = max(j - 1, 1)
#     j0 = min(j, ny)
#     vx = (ϕ.Vx[i, j0] > 0) * (ϕ.Vx[i, j_bot] > 0)

#     nx, ny = size(ϕ.Vy)
#     i_left = max(i - 1, 1)
#     i0 = min(i, nx)
#     vy = (ϕ.Vy[i0, j] > 0) * (ϕ.Vy[i_left, j] > 0)
#     v = vx * vy
#     return v * (ϕ.vertex[i, j] > 0)
# end

# """
#     isvalid_vx(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.Vx[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_vx(ϕ::JustRelax.RockRatio, i, j)
#     # c = (ϕ.center[i, j] > 0) * (ϕ.center[i - 1, j] > 0)
#     # v = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i, j + 1] > 0)
#     # cv = c * v
#     # return cv * (ϕ.Vx[i, j] > 0)
#     return (ϕ.Vx[i, j] > 0)
# end

# """
#     isvalid_vy(ϕ::JustRelax.RockRatio, inds...)

# Check if  `ϕ.Vy[inds...]` is a not a nullspace.

# # Arguments
# - `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
# - `inds`: Cartesian indices to check.
# """
# Base.@propagate_inbounds @inline function isvalid_vy(ϕ::JustRelax.RockRatio, i, j)
#     # c = (ϕ.center[i, j] > 0) * (ϕ.center[i, j - 1] > 0)
#     # v = (ϕ.vertex[i, j] > 0) * (ϕ.vertex[i + 1, j] > 0)
#     # cv = c * v
#     # return cv * (ϕ.Vy[i, j] > 0)
#     return (ϕ.Vy[i, j] > 0)
# end

#######

"""
    isvalid_c(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.center[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j)
    (ϕ.center[i, j] > 0)
end

"""
    isvalid_v(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.vertex[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_v(ϕ::JustRelax.RockRatio, i, j)
    (ϕ.vertex[i, j] > 0)
end

"""
    isvalid_vx(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.Vx[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_vx(ϕ::JustRelax.RockRatio, i, j)
    c = (ϕ.center[i, j] > 0) || (ϕ.center[i - 1, j] > 0)
    v = (ϕ.vertex[i, j] > 0) || (ϕ.vertex[i, j + 1] > 0)
    cv = c || v
    return cv || (ϕ.Vx[i, j] > 0)
end

"""
    isvalid_vy(ϕ::JustRelax.RockRatio, inds...)

Check if  `ϕ.Vy[inds...]` is a not a nullspace.

# Arguments
- `ϕ::JustRelax.RockRatio`: The `RockRatio` object to check against.
- `inds`: Cartesian indices to check.
"""
Base.@propagate_inbounds @inline function isvalid_vy(ϕ::JustRelax.RockRatio, i, j)
    c = (ϕ.center[i, j] > 0) || (ϕ.center[i, j - 1] > 0)
    v = (ϕ.vertex[i, j] > 0) || (ϕ.vertex[i + 1, j] > 0)
    cv = c || v
    return cv || (ϕ.Vy[i, j] > 0)
end

Base.@propagate_inbounds @inline isvalid(ϕ, I::Vararg{Integer, N}) where N = ϕ[I...] > 0
