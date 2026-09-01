function RockRatio(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return RockRatio(ni...)
end

function RockRatio(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return RockRatio(ni...)
end

"""
    RockRatio(nx, ny)

Create a `RockRatio` object for a 2D grid with dimensions `nx` x `ny` on a staggered grid.

"""
function RockRatio(nx, ny)
    # The four fields below are liquid-volume fractions, not material
    # properties. Their staggered locations are:
    #
    #                  Vy (nx, ny+1)
    #         Vx       center       Vx
    #                  vertex (nx+1, ny+1)
    ni = nx, ny
    center = @zeros(ni...)
    vertex = @zeros(ni .+ 1...)
    Vx = @zeros(nx + 1, ny) # no ghost nodes!
    Vy = @zeros(nx, ny + 1) # no ghost nodes!
    dummy = @zeros(1, 1) # because it cant be a Union{T, Nothing} type on the GPU....
    return JustRelax.RockRatio(center, vertex, Vx, Vy, dummy, dummy, dummy, dummy)
end

"""
    RockRatio(nx, ny, nz)

Create a `RockRatio` object for a 3D grid with dimensions `nx` x `ny` x `nz` on a staggered grid.
"""
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

# A rock-ratio entry is active when it carries any liquid volume.  Keep this
# policy in one place: all staggered-grid null-space checks below use it, so a
# positive sub-cell fraction is not silently treated as an inactive DOF by one
# operator and active by another.
@inline variational_active(x) = x > zero(x)

# Bounded approximation of the inverse liquid face mass.  The positive floor
# keeps partially filled velocity rows finite; fully filled rows remain exact.
@inline variational_face_mass(x) = max(x, oftype(x, 0.1))

"""
    update_rock_ratio!(ϕ::JustRelax.RockRatio, phase_ratios, air_phase)

Update the rock ratio `ϕ` based on the provided `phase_ratios` and `air_phase`.

# Arguments
- `ϕ::JustRelax.RockRatio`: The rock ratio object to be updated.
- `phase_ratios`: The ratios of different phases present.
- `air_phase`: The phase representing air.
"""
function update_rock_ratio!(ϕ::JustRelax.RockRatio{T, 2}, phase_ratios, air_phase) where {T}
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

"""
    update_rock_ratio!(ϕ::JustRelax.RockRatio, phase_ratios, air_phase)

Update the rock ratio `ϕ` for a 3D grid based on the provided `phase_ratios` and `air_phase`.

# Arguments
- `ϕ::JustRelax.RockRatio`: The rock ratio object to be updated.
- `phase_ratios`: The ratios of different phases present.
- `air_phase`: The phase representing air.
"""
function update_rock_ratio!(ϕ::JustRelax.RockRatio{T, 3}, phase_ratios, air_phase) where {T}
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

"""
    compute_rock_ratio(phase_ratio, air_phase, inds...)

Compute the rock ratio at the given indices based on the `phase_ratio` and `air_phase`.
"""
@inline function compute_rock_ratio(
        phase_ratio::CellArray, air_phase, I::Vararg{Integer, N}
    ) where {N}
    1 ≤ air_phase ≤ numphases(phase_ratio) || return 1.0e0
    x = 1 - @index phase_ratio[air_phase, I...]
    x *= x > 1.0e-5
    return x
end

"""
    compute_air_ratio(phase_ratio, air_phase, inds...)

Compute the air ratio at the given indices based on the `phase_ratio` and `air_phase`.
"""
@inline function compute_air_ratio(
        phase_ratio::CellArray, air_phase, I::Vararg{Integer, N}
    ) where {N}
    1 ≤ air_phase ≤ numphases(phase_ratio) || return 1.0e0
    return @index phase_ratio[air_phase, I...]
end

"""
    update_rock_ratio_cv!(ϕ, ratio_center, ratio_vertex, air_phase)

Update the rock ratio for both center and vertex values based on the provided `ratio_center`, `ratio_vertex`, and `air_phase`.
"""
@parallel_indices (I...) function update_rock_ratio_cv!(
        ϕ, ratio_center, ratio_vertex, air_phase
    )
    if all(I .≤ size(ratio_center))
        ϕ.center[I...] = compute_rock_ratio(ratio_center, air_phase, I...)
    end
    ϕ.vertex[I...] = compute_rock_ratio(ratio_vertex, air_phase, I...)
    return nothing
end

"""
    _update_rock_ratio!(ϕ, ratio, air_phase)

Inner kernel of `update_rock_ratio` that clamps the computed rock ratio to the range [0, 1] for the given `ratio` and `air_phase`.
"""
@parallel_indices (I...) function _update_rock_ratio!(ϕ, ratio, air_phase)
    # ϕ[I...] = Float64(Float16(compute_rock_ratio(ratio, air_phase, I...)))
    ϕ[I...] = clamp(compute_rock_ratio(ratio, air_phase, I...), 0, 1)
    return nothing
end

"""
    isvalid_c(ϕ::JustRelax.RockRatio, inds...)

Check whether the 2D pressure degree of freedom is connected to liquid.

The cell-centred pressure row is retained only when `ϕ.center[i,j]` and all
four adjacent velocity faces are active:

                 Vy[i, j+1]
                       o
                       |
        Vx[i, j]  o--- p[i,j] ---o  Vx[i+1, j]
                       |
                       o
                 Vy[i, j]

This is the local null-space elimination used by the matrix-free reduced
system. It is intentionally stricter than testing the centre fraction alone.

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `i`, `j`: cell indices.
"""
Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j)
    # A pressure cell is retained only when its centre and all four adjacent
    # velocity faces are connected to liquid:
    #
    #                  Vy[i, j+1]
    #                        o
    #                        |
    #         Vx[i, j]  o--- p[i,j] ---o  Vx[i+1, j]
    #                        |
    #                        o
    #                  Vy[i, j]
    vx = isvalid(ϕ.Vx, i, j) * isvalid(ϕ.Vx[i + 1, j])
    vy = isvalid(ϕ.Vy, i, j) * isvalid(ϕ.Vy[i, j + 1])
    v = vx * vy
    return v * isvalid(ϕ.center, i, j)
end

"""
    isvalid_c(ϕ::JustRelax.RockRatio, i, j, k)

Whether the pressure at `ϕ.center[i, j, k]` belongs to the reduced space in 3D.

As in 2D, the center and all six faces read by the divergence constraint must belong to the
reduced space.
"""
Base.@propagate_inbounds @inline function isvalid_c(ϕ::JustRelax.RockRatio, i, j, k)
    vx = isvalid(ϕ.Vx, i, j, k) * isvalid(ϕ.Vx, i + 1, j, k)
    vy = isvalid(ϕ.Vy, i, j, k) * isvalid(ϕ.Vy, i, j + 1, k)
    vz = isvalid(ϕ.Vz, i, j, k) * isvalid(ϕ.Vz, i, j, k + 1)
    return vx * vy * vz * isvalid(ϕ.center, i, j, k)
end

"""
    update_valid_v_masks!(maskVx, maskVy, ϕ)

Fill the 2D velocity masks with the exact reduced space used by the variational momentum
kernels. The masks have residual-array shapes: `maskVx[i, j]` corresponds to `ϕ.Vx[i+1, j]`,
and `maskVy[i, j]` to `ϕ.Vy[i, j+1]`.

Validity includes the velocity-face fraction and every center and shear-stress vertex read by
the corresponding momentum row. Using these masks for residual norms and Rayleigh quotients
keeps the measured system identical to the system being iterated.
"""
@parallel_indices (i, j) function update_valid_v_masks!(maskVx, maskVy, ϕ::JustRelax.RockRatio)
    if i ≤ size(maskVx, 1) && j ≤ size(maskVx, 2)
        maskVx[i, j] = isvalid_vx_strict(ϕ, i + 1, j)
    end
    if i ≤ size(maskVy, 1) && j ≤ size(maskVy, 2)
        maskVy[i, j] = isvalid_vy_strict(ϕ, i, j + 1)
    end
    return nothing
end

"""
    project_reduced_state!(P, P0, ΔPψ, λ, Vx, Vy, ϕ)

Project carried 2D Stokes state onto the reduced space defined by `ϕ`. Eliminated pressure-like
unknowns and interior velocities are set to zero. The operation is idempotent and is intended to
run before each variational solve, because marker or interface advection can change the mask
between time steps.

Dynamic-relaxation history is not part of this function and must be reset separately whenever
the operator is rebuilt.
"""
@parallel_indices (i, j) function project_reduced_state!(P, P0, ΔPψ, λ, Vx, Vy, ϕ::JustRelax.RockRatio)
    if i ≤ size(P, 1) && j ≤ size(P, 2) && !isvalid_c(ϕ, i, j)
        P[i, j] = zero(eltype(P))
        P0[i, j] = zero(eltype(P0))
        ΔPψ[i, j] = zero(eltype(ΔPψ))
        λ[i, j] = zero(eltype(λ))
    end
    if i ≤ size(Vx, 1) - 2 && j ≤ size(Vx, 2) - 2 && !isvalid_vx_strict(ϕ, i + 1, j)
        Vx[i + 1, j + 1] = zero(eltype(Vx))
    end
    if i ≤ size(Vy, 1) - 2 && j ≤ size(Vy, 2) - 2 && !isvalid_vy_strict(ϕ, i, j + 1)
        Vy[i + 1, j + 1] = zero(eltype(Vy))
    end
    return nothing
end

"""
    isvalid_v(ϕ::JustRelax.RockRatio, i, j)

Whether the shear strain rate / stress at `ϕ.vertex[i, j]` belongs to the reduced space in 2D.

`εxy` at a vertex is built from the velocity gradients across it, so the vertex needs rock and
so do the two `Vx` faces below and above it and the two `Vy` faces left and right of it. The
indices are clamped, so a vertex on the domain edge is judged on the faces that exist.

                Vx[i,j]
                   │
        Vy[i-1,j] ─●─ Vy[i,j]
                   │   vertex[i,j]
                Vx[i,j-1]

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `i`, `j`: vertex indices.
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

Whether `Vx[inds...]` carries rock on its own face.

This tests the face fraction alone and says nothing about the cells and vertices the x-momentum
row reads; [`isvalid_vx_strict`](@ref) adds that condition.

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `inds`: face indices.
"""
Base.@propagate_inbounds @inline function isvalid_vx(
        ϕ::JustRelax.RockRatio, I::Vararg{Integer, N}
    ) where {N}
    return isvalid(ϕ.Vx, I...)
end

"""
    isvalid_vx_strict(ϕ::JustRelax.RockRatio, i, j)

Whether `Vx[i, j]` belongs to the reduced space in 2D.

A face fraction alone does not make the x-momentum row solvable. The row reads `τxx` and `P` at
the two centers the face separates and `τxy` at the two vertices it spans, so all four have to
carry rock as well — a face that keeps its equation while any of them is void is the velocity
counterpart of the pressure null space [`isvalid_c`](@ref) removes.

                  ● vertex[i,j+1]
                  │
    center[i-1,j] │ Vx[i,j] │ center[i,j]
                  │
                  ● vertex[i,j]

Such a face is free to drift: every coefficient of its row is scaled by a rock fraction near
zero, so the velocity there can grow far beyond the physical field while the residual stays
small and the masked norms report convergence. The indices are clamped, so a face on the domain
edge is judged on the stencil that exists.

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `i`, `j`: face indices.
"""
Base.@propagate_inbounds @inline function isvalid_vx_strict(ϕ::JustRelax.RockRatio, i::Integer, j::Integer)
    nxc, nyc = size(ϕ.center)
    c = isvalid(ϕ.center, clamp(i - 1, 1, nxc), clamp(j, 1, nyc)) *
        isvalid(ϕ.center, clamp(i, 1, nxc), clamp(j, 1, nyc))
    nxv, nyv = size(ϕ.vertex)
    v = isvalid(ϕ.vertex, clamp(i, 1, nxv), clamp(j, 1, nyv)) *
        isvalid(ϕ.vertex, clamp(i, 1, nxv), clamp(j + 1, 1, nyv))
    return c * v * isvalid(ϕ.Vx, i, j)
end

"""
    isvalid_vy(ϕ::JustRelax.RockRatio, inds...)

Whether `Vy[inds...]` carries rock on its own face.

This tests the face fraction alone and says nothing about the cells and vertices the y-momentum
row reads; [`isvalid_vy_strict`](@ref) adds that condition.

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `inds`: face indices.
"""
Base.@propagate_inbounds @inline function isvalid_vy(
        ϕ::JustRelax.RockRatio, I::Vararg{Integer, N}
    ) where {N}
    return isvalid(ϕ.Vy, I...)
end

"""
    isvalid_vy_strict(ϕ::JustRelax.RockRatio, i, j)

Whether `Vy[i, j]` belongs to the reduced space in 2D.

Transpose of [`isvalid_vx_strict`](@ref): the y-momentum row reads `τyy` and `P` at the centers
below and above the face, and `τxy` at the vertices to either side of it.

                  center[i,j]
    vertex[i,j] ●──── Vy[i,j] ────● vertex[i+1,j]
                 center[i,j-1]

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `i`, `j`: face indices.
"""
Base.@propagate_inbounds @inline function isvalid_vy_strict(ϕ::JustRelax.RockRatio, i::Integer, j::Integer)
    nxc, nyc = size(ϕ.center)
    c = isvalid(ϕ.center, clamp(i, 1, nxc), clamp(j - 1, 1, nyc)) *
        isvalid(ϕ.center, clamp(i, 1, nxc), clamp(j, 1, nyc))
    nxv, nyv = size(ϕ.vertex)
    v = isvalid(ϕ.vertex, clamp(i, 1, nxv), clamp(j, 1, nyv)) *
        isvalid(ϕ.vertex, clamp(i + 1, 1, nxv), clamp(j, 1, nyv))
    return c * v * isvalid(ϕ.Vy, i, j)
end

"""
    isvalid_vz(ϕ::JustRelax.RockRatio, inds...)

Whether `Vz[inds...]` carries rock on its own face.

# Arguments
- `ϕ::JustRelax.RockRatio`: rock fractions on the staggered grid.
- `inds`: face indices.
"""
Base.@propagate_inbounds @inline function isvalid_vz(
        ϕ::JustRelax.RockRatio, I::Vararg{Integer, N}
    ) where {N}
    return isvalid(ϕ.Vz, I...)
end

"""
    isvalid_velocity(ϕ::JustRelax.RockRatio, i, j)

Whether both velocity faces indexed `(i, j)` carry rock, in 2D.

Note that `Vx[i, j]` and `Vy[i, j]` are the west and south faces of the same cell rather than
one location, so this is a joint test on the pair, not on a single unknown.
"""
Base.@propagate_inbounds @inline function isvalid_velocity(ϕ::JustRelax.RockRatio, i, j)
    return isvalid(ϕ.Vx, i, j) * isvalid(ϕ.Vy, i, j)
end

"""
    isvalid_velocity(ϕ::JustRelax.RockRatio, i, j, k)

Whether all three velocity faces indexed `(i, j, k)` carry rock, in 3D.
"""
Base.@propagate_inbounds @inline function isvalid_velocity(ϕ::JustRelax.RockRatio, i, j, k)
    return isvalid(ϕ.Vx, i, j, k) * isvalid(ϕ.Vy, i, j, k) * isvalid(ϕ.Vz, i, j, k)
end

"""
    isvalid_v(ϕ::JustRelax.RockRatio, i, j, k)

Whether the corner at `ϕ.vertex[i, j, k]` belongs to the reduced space in 3D.

The 3D counterpart of the 2D vertex test: the corner needs rock, and so do the shear locations
meeting at it — `yz` on either side in x, `xz` on either side in y and `xy` on either side in z.
Indices are clamped, so a corner on the domain boundary is judged on what exists.
"""
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
    xy = isvalid(ϕ.xy, i, j, k_top) * isvalid(ϕ.xy, i, j, k_bot)

    # V
    v = yz * xz * xy

    return v * isvalid(ϕ.vertex, i, j, k)
end

"""
    isvalid_xz(ϕ::JustRelax.RockRatio, i, j, k)

Whether the `xz` shear location at `(i, j, k)` belongs to the reduced space.

`εxz` is built from `∂Vx/∂z` and `∂Vz/∂x`, so it needs the two vertices bounding it in y, the
two `Vz` faces on either side in x and the two `Vx` faces on either side in z.
"""
Base.@propagate_inbounds @inline function isvalid_xz(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i, j + 1, k)

    # check vz
    nx, ny, nz = size(ϕ.Vz)
    i_left = max(i - 1, 1)
    i_right = min(i, nx)
    vz = isvalid(ϕ.Vz, i_left, j, k) * isvalid(ϕ.Vz, i_right, j, k)

    # check vx
    nx, ny, nz = size(ϕ.Vx)
    k_top = max(k - 1, 1)
    k_bot = min(k, nz)
    vx = isvalid(ϕ.Vx, i, j, k_top) * isvalid(ϕ.Vx, i, j, k_bot)

    return v * vx * vz * isvalid(ϕ.vertex, i, j, k)
end

"""
    isvalid_xy(ϕ::JustRelax.RockRatio, i, j, k)

Whether the `xy` shear location at `(i, j, k)` belongs to the reduced space.

`εxy` is built from `∂Vx/∂y` and `∂Vy/∂x`, so it needs the two vertices bounding it in z, the
two `Vx` faces on either side in y and the two `Vy` faces on either side in x.
"""
Base.@propagate_inbounds @inline function isvalid_xy(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i, j, k + 1)

    # check vx
    nx, ny, nz = size(ϕ.Vx)
    j_front = max(j - 1, 1)
    j_back = min(j, ny)
    vx = isvalid(ϕ.Vx, i, j_front, k) * isvalid(ϕ.Vx, i, j_back, k)

    # check vy
    nx, ny, nz = size(ϕ.Vy)
    i_left = max(i - 1, 1)
    i_right = min(i, nx)
    vy = isvalid(ϕ.Vy, i_left, j, k) * isvalid(ϕ.Vy, i_right, j, k)

    return v * vx * vy * isvalid(ϕ.vertex, i, j, k)
end

"""
    isvalid_yz(ϕ::JustRelax.RockRatio, i, j, k)

Whether the `yz` shear location at `(i, j, k)` belongs to the reduced space.

`εyz` is built from `∂Vy/∂z` and `∂Vz/∂y`, so it needs the two vertices bounding it in x, the
two `Vz` faces on either side in y and the two `Vy` faces on either side in z.
"""
Base.@propagate_inbounds @inline function isvalid_yz(ϕ::JustRelax.RockRatio, i, j, k)

    # check vertices
    v = isvalid(ϕ.vertex, i, j, k) * isvalid(ϕ.vertex, i + 1, j, k)

    # check vz
    nx, ny, nz = size(ϕ.Vz)
    j_front = max(j - 1, 1)
    j_back = min(j, ny)
    vz = isvalid(ϕ.Vz, i, j_front, k) * isvalid(ϕ.Vz, i, j_back, k)

    # check vy
    nx, ny, nz = size(ϕ.Vy)
    k_top = max(k - 1, 1)
    k_bot = min(k, nz)
    vy = isvalid(ϕ.Vy, i, j, k_top) * isvalid(ϕ.Vy, i, j, k_bot)

    return v * vy * vz * isvalid(ϕ.vertex, i, j, k)
end

Base.@propagate_inbounds @inline isvalid(ϕ, I::Vararg{Integer, N}) where {N} =
    variational_active(ϕ[I...])

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
