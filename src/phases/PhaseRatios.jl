"""
    update_phase_ratios_2D!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractMatrix}, xci, xvi
    )

JustRelax routine based on `JustPIC.update_phase_ratios!`.
Update the center, vertex and velocity-face phase ratios in `phase_ratios` from the
2-D `phase_arrays`, given the cell-center coordinates `xci` and vertex coordinates `xvi`.
The phase arrays need to be AbstractArrays and have values between 0 and 1.

#Example:
```julia
nx, ny = 100, 100
phase_1 = zeros(nx, ny)
phase_1[User_criterion .== true] .= 1.0
phase_2 = zeros(nx, ny)
phase_2[User_criterion .== false] .= 1.0
phase_arrays = (phase_1, phase_2)

# Advect both phase arrays and update phase ratios
update_phase_ratios_2D!(phase_ratios, phase_arrays, xci, xvi)
```
"""
function update_phase_ratios_2D!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, phase_arrays::NTuple{N, AbstractMatrix}, xci, xvi
    ) where {B, T <: AbstractMatrix, N}

    phase_ratios_center_from_arrays!(phase_ratios, phase_arrays)
    phase_ratios_vertex_from_arrays!(phase_ratios, phase_arrays, xvi, xci)
    # velocity nodes
    phase_ratios_face_from_arrays!(phase_ratios.Vx, phase_arrays, xci, :x)
    phase_ratios_face_from_arrays!(phase_ratios.Vy, phase_arrays, xci, :y)

    return nothing
end

"""
    update_phase_ratios_3D!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}, xci, xvi
    )

JustRelax routine based on `JustPIC.update_phase_ratios!`.
Update the center, vertex, velocity-face and shear-stress-midpoint phase ratios in
`phase_ratios` from the 3-D `phase_arrays`, given the cell-center coordinates `xci` and
vertex coordinates `xvi`.
The phase arrays need to be AbstractArrays and have values between 0 and 1.

#Example:
```julia
nx, ny, nz = 100, 100, 100
phase_1 = zeros(nx, ny, nz)
phase_1[User_criterion .== true] .= 1.0
phase_2 = zeros(nx, ny, nz)
phase_2[User_criterion .== false] .= 1.0
phase_arrays = (phase_1, phase_2)

# Advect both phase arrays and update phase ratios
update_phase_ratios_3D!(phase_ratios, phase_arrays, xci, xvi)
```
"""
function update_phase_ratios_3D!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, phase_arrays::NTuple{N, AbstractArray}, xci, xvi
    ) where {B, T <: AbstractArray, N}

    phase_ratios_center_from_arrays!(phase_ratios, phase_arrays)
    phase_ratios_vertex_from_arrays!(phase_ratios, phase_arrays, xvi, xci)

    # velocity nodes
    phase_ratios_face_from_arrays!(phase_ratios.Vx, phase_arrays, xci, :x)
    phase_ratios_face_from_arrays!(phase_ratios.Vy, phase_arrays, xci, :y)
    phase_ratios_face_from_arrays!(phase_ratios.Vz, phase_arrays, xci, :z)

    # shear stress nodes
    phase_ratios_midpoint_from_arrays!(phase_ratios.xy, phase_arrays, xci, :xy)
    phase_ratios_midpoint_from_arrays!(phase_ratios.yz, phase_arrays, xci, :yz)
    phase_ratios_midpoint_from_arrays!(phase_ratios.xz, phase_arrays, xci, :xz)
    return nothing
end

function phase_ratios_center_from_arrays!(phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}) where {N}
    ni = size(first(phase_arrays))

    @parallel (@idx ni) phase_ratios_center_from_arrays_kernel!(
        phase_ratios.center, phase_arrays
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_from_arrays_kernel!(
        ratio_centers, phase_arrays::NTuple{N, AbstractArray}
    ) where {N}

    values = clamp.(ntuple(i -> phase_arrays[i][I...], Val(N)), 0, 1)
    # Sum
    total = sum(values)

    # Normalize
    normalized = values ./ total

    # Clamp
    clamped = clamp.(normalized, 0.0, 1.0)

    # Threshold small values
    cleaned = map(x -> x < 1.0e-5 ? zero(eltype(values)) : x, clamped)

    # Renormalize
    final_total = sum(cleaned)
    final = cleaned ./ final_total

    # Write back
    for k in 1:N
        @index ratio_centers[k, I...] = final[k]
    end

    return nothing
end

## ============================================================================
## VERTEX VALUES
## ============================================================================

function phase_ratios_vertex_from_arrays!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}, xvi::NTuple{ND}, xci::NTuple{ND}
    ) where {N, ND}

    ni = size(first(phase_arrays)) .+ 1
    di = JustPIC.compute_dx(xvi)

    @parallel (@idx ni) phase_ratios_vertex_from_arrays_kernel!(
        phase_ratios.vertex, phase_arrays, xci, xvi, di
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_from_arrays_kernel!(
        ratio_vertices, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, xvi::NTuple{ND}, di::NTuple{ND, T}
    ) where {N, ND, T}
    w_vals = @MVector zeros(T, N)
    total_weight = zero(T)

    # Vertex position
    cell_vertex = ntuple(d -> xvi[d][I[d]], Val(ND))
    ni = size(first(phase_arrays))

    if ND == 2
        for offset₁ in -1:0, offset₂ in -1:0
            i_cell = I[1] + offset₁
            j_cell = I[2] + offset₂
            if 1 ≤ i_cell ≤ ni[1] && 1 ≤ j_cell ≤ ni[2]
                x_c = xci[1][i_cell]
                y_c = xci[2][j_cell]
                dx, dy = @dxi(di, i_cell, j_cell)
                wx = muladd(-abs(cell_vertex[1] - x_c), inv(dx), 1.0)
                wy = muladd(-abs(cell_vertex[2] - y_c), inv(dy), 1.0)
                weight = wx * wy
                total_weight += weight

                # Clamp so the accumulated values stay non-negative
                for k in 1:N
                    @inbounds w_vals[k] += weight * clamp(phase_arrays[k][i_cell, j_cell], zero(T), one(T))
                end
            end
        end

    elseif ND == 3
        vertex_pos = ntuple(d -> xvi[d][I[d]], Val(ND))
        for offset₁ in -1:0, offset₂ in -1:0, offset₃ in -1:0
            i_cell = I[1] + offset₁
            j_cell = I[2] + offset₂
            k_cell = I[3] + offset₃
            if 1 ≤ i_cell ≤ ni[1] && 1 ≤ j_cell ≤ ni[2] && 1 ≤ k_cell ≤ ni[3]
                cell_center = (xci[1][i_cell], xci[2][j_cell], xci[3][k_cell])
                dx, dy, dz = @dxi(di, i_cell, j_cell, k_cell)
                dxyz = (dx, dy, dz)
                # Use trilinear weights for 3D interpolation
                weight = 1.0
                for d in 1:3
                    weight *= (1.0 - abs(vertex_pos[d] - cell_center[d]) * inv(dxyz[d]))
                end
                total_weight += weight

                # Clamp so the accumulated values stay non-negative
                for k in 1:N
                    @inbounds w_vals[k] += weight * clamp(phase_arrays[k][i_cell, j_cell, k_cell], zero(T), one(T))
                end
            end
        end
    end

    # Normalize
    @inbounds for k in 1:N
        w_vals[k] /= total_weight
    end

    # Clamp
    @inbounds for k in 1:N
        w_vals[k] = clamp(w_vals[k], zero(T), one(T))
    end

    # Threshold small values and renormalize
    total = zero(T)
    @inbounds for k in 1:N
        w_vals[k] = w_vals[k] < T(1.0e-5) ? zero(T) : w_vals[k]
        total += w_vals[k]
    end

    @inbounds for k in 1:N
        w_vals[k] /= total
        @index ratio_vertices[k, I...] = w_vals[k]
    end

    return nothing
end

## ============================================================================
## FACE VALUES
## ============================================================================

function phase_ratios_face_from_arrays!(
        phase_face, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, dimension::Symbol
    ) where {N, ND}
    ni = size(first(phase_arrays))  # Cell grid size
    di = JustPIC.compute_dx(xci)
    offsets = JustPIC.face_offset(Val(ND), dimension)
    face_ni = ntuple(d -> ni[d] + offsets[d], Val(ND))

    @parallel (@idx face_ni) phase_ratios_face_from_arrays_kernel!(
        phase_face, phase_arrays, xci, di, offsets, ni
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_face_from_arrays_kernel!(
        ratio_faces, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, di::NTuple{ND, T}, offsets, ni
    ) where {N, ND, T}

    w_vals = @MVector zeros(T, N)
    total_weight = zero(T)
    for side in 0:1
        # Calculate which cell this face point samples from
        cell_index = ntuple(
            d -> begin
                if offsets[d] == 1  # This dimension is staggered
                    # Face I[d] is between cells I[d]-1 and I[d]
                    # side=0 gives left cell (I[d]-1), side=1 gives right cell (I[d])
                    I[d] - 1 + side
                else
                    # Non-staggered dimension: use face index directly
                    I[d]
                end
            end, Val(ND)
        )

        # Check if cell index is within bounds
        valid_cell = all(1 ≤ cell_index[d] ≤ ni[d] for d in 1:ND)
        !valid_cell && continue

        # Equal weighting from both adjacent cells
        weight = T(0.5)
        total_weight += weight

        # Accumulate weighted phase values from this cell, clamped so the
        # accumulated values stay non-negative
        @inbounds for k in 1:N
            w_vals[k] += weight * clamp(phase_arrays[k][cell_index...], zero(T), one(T))
        end
    end

    # Normalize
    @inbounds for k in 1:N
        w_vals[k] /= total_weight
    end

    # Clamp
    @inbounds for k in 1:N
        w_vals[k] = min(max(w_vals[k], zero(T)), one(T))
    end

    # Clean up very small values and renormalize
    total = zero(T)
    @inbounds for k in 1:N
        w_vals[k] = w_vals[k] < T(1.0e-5) ? zero(T) : w_vals[k]
        total += w_vals[k]
    end

    @inbounds for k in 1:N
        w_vals[k] /= total
    end

    # Write to face grid
    @inbounds for ip in 1:N
        @index ratio_faces[ip, I...] = w_vals[ip]
    end

    return nothing
end

## ============================================================================
## MIDPOINT VALUES
## ============================================================================

function phase_ratios_midpoint_from_arrays!(
        phase_midpoints, phase_arrays::NTuple{N, AbstractArray}, xci, dimension
    ) where {N}
    ni = size(first(phase_arrays))  # Cell grid size
    di = JustPIC.compute_dx(xci)

    # Define staggered offsets for midpoint grids
    offsets = if dimension === :xy
        (1, 1, 0)  # Staggered in x and y
    elseif dimension === :yz
        (0, 1, 1)  # Staggered in y and z
    elseif dimension === :xz
        (1, 0, 1)  # Staggered in x and z
    else
        throw("Unknown dimension: $(dimension). Valid dimensions are :xy, :yz, :xz")
    end

    midpoint_ni = ntuple(d -> ni[d] + offsets[d], Val(length(ni)))

    @parallel (@idx midpoint_ni) phase_ratios_midpoint_from_arrays_kernel!(
        phase_midpoints, phase_arrays, xci, di, offsets, ni
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_midpoint_from_arrays_kernel!(
        ratio_midpoints, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{3}, di::NTuple{3, T}, offsets, ni
    ) where {N, T}

    w_vals = @MVector zeros(T, N)
    total_weight = zero(T)

    # Every midpoint grid is staggered in exactly two dimensions. Spell out the
    # four neighbouring cells: mutating a captured bit counter in an `ntuple`
    # closure boxes the counter, which is unsupported in GPU kernels.
    for first_offset in -1:0, second_offset in -1:0
        i_cell = I[1] + offsets[1] * first_offset
        j_offset = ifelse(offsets[1] == 1, second_offset, first_offset)
        j_cell = I[2] + offsets[2] * j_offset
        k_cell = I[3] + offsets[3] * second_offset

        if !(1 ≤ i_cell ≤ ni[1] && 1 ≤ j_cell ≤ ni[2] && 1 ≤ k_cell ≤ ni[3])
            continue
        end

        # Equal weighting from all corner cells
        weight = T(0.25)
        total_weight += weight

        # Accumulate weighted phase values from this corner cell, clamped so
        # the accumulated values stay non-negative
        @inbounds for k in 1:N
            w_vals[k] += weight * clamp(
                phase_arrays[k][i_cell, j_cell, k_cell], zero(T), one(T)
            )
        end
    end

    # Normalize
    @inbounds for k in 1:N
        w_vals[k] /= total_weight
    end

    # Clamp
    @inbounds for k in 1:N
        w_vals[k] = min(max(w_vals[k], zero(T)), one(T))
    end

    # Clean up very small values and renormalize
    total = zero(T)
    @inbounds for k in 1:N
        w_vals[k] = w_vals[k] < T(1.0e-5) ? zero(T) : w_vals[k]
        total += w_vals[k]
    end

    @inbounds for k in 1:N
        w_vals[k] /= total
    end

    # Write to midpoint grid
    @inbounds for ip in 1:N
        @index ratio_midpoints[ip, I...] = w_vals[ip]
    end

    return nothing
end
