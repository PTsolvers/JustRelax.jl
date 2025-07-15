"""
    update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractMatrix}, xci, xvi
    ) where {B, T <: AbstractMatrix, N}

JustRelax routine based on `JustPIC._2D.update_phase_ratios!` or `JustPIC._3D.update_phase_ratios!`.
Update the phase ratios in `phase_ratios` using the provided `phase_arrays`, `xci`, and `xvi`.
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
update_phase_ratios!(phase_ratios, phase_arrays, xci, xvi)
```
"""
function update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, phase_arrays::NTuple{N, AbstractMatrix}, xci, xvi
    ) where {B, T <: AbstractMatrix, N}


    phase_ratios_center_from_arrays!(phase_ratios, phase_arrays, xci)
    phase_ratios_vertex_from_arrays!(phase_ratios, phase_arrays, xvi, xci)
    # velocity nodes
    phase_ratios_face_from_arrays!(phase_ratios.Vx, phase_arrays, xci, :x)
    phase_ratios_face_from_arrays!(phase_ratios.Vy, phase_arrays, xci, :y)

    return nothing
end

"""
    update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}, xci, xvi
    ) where {B, T <: AbstractArray, N}

JustRelax routine based on `JustPIC._2D.update_phase_ratios!` or `JustPIC._3D.update_phase_ratios!`.
Update the phase ratios in `phase_ratios` using the provided `phase_arrays`, `xci`, and `xvi`.
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
update_phase_ratios!(phase_ratios, phase_arrays, xci, xvi)
```
"""
function update_phase_ratios!(
        phase_ratios::JustPIC.PhaseRatios{B, T}, phase_arrays::NTuple{N, AbstractArray}, xci, xvi
    ) where {B, T <: AbstractArray, N}

    phase_ratios_center_from_arrays!(phase_ratios, phase_arrays, xci)
    phase_ratios_vertex_from_arrays!(phase_ratios, phase_arrays, xvi, xci)

    # velocity nodes
    phase_ratios_face_from_arrays!(phase_ratios.Vx, phase_arrays, xci, :x)
    phase_ratios_face_from_arrays!(phase_ratios.Vy, phase_arrays, xci, :y)
    phase_ratios_face_from_arrays!(phase_ratios.Vz, phase_arrays, xci, :z)

    # shear stress nodes
    phase_ratios_midpoint_from_arrays!(phase_ratios.xy, phase_arrays, xci, :xy)
    phase_ratios_midpoint_from_arrays!(phase_ratios.yz, phase_arrays, xci, :yz)
    phase_ratios_midpoint_from_arrays!(phase_ratios.xz, phase_arrays, xci, :xz)
    end
    return nothing
end

function phase_ratios_center_from_arrays!(phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}, xci) where {N}
    ni = size(first(phase_arrays))
    di = compute_dx(xci)

    @parallel (@idx ni) phase_ratios_center_from_arrays_kernel!(
        phase_ratios.center, phase_arrays, xci, di
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_center_from_arrays_kernel!(
        ratio_centers, phase_arrays::NTuple{N, AbstractArray}
    ) where {N}

    values = ntuple(i -> phase_arrays[i][I...], Val(N))
    total = sum(values)

    # Normalize (handle case where total might be zero)
    if total > eps(typeof(total))
        normalized_values = clamp.(values ./ total, 0.0, 1.0)
        # Clean up very small values (round to zero if < 1e-5)
        cleaned_values = ntuple(i -> normalized_values[i] < 1e-5 ? zero(eltype(normalized_values)) : normalized_values[i], Val(N))
        # Renormalize to ensure sum = 1
        final_total = sum(cleaned_values)
        if final_total > eps(typeof(final_total))
            normalized_values = cleaned_values ./ final_total
        else
            normalized_values = ntuple(_ -> one(eltype(values)) / N, Val(N))
        end
    else
        # If all values are zero, distribute equally
        normalized_values = ntuple(_ -> one(eltype(values)) / N, Val(N))
    end

    # Update phase ratios array
    for k in 1:N
        @index ratio_centers[k, I...] = normalized_values[k]
    end

    return nothing
end


function phase_ratios_vertex_from_arrays!(
        phase_ratios::JustPIC.PhaseRatios, phase_arrays::NTuple{N, AbstractArray}, xvi::NTuple{ND}, xci::NTuple{ND}
    ) where {N, ND}

    ni = size(first(phase_arrays)) .+1
    di = compute_dx(xvi)

    @parallel (@idx ni) phase_ratios_vertex_from_arrays_kernel!(
        phase_ratios.vertex, phase_arrays, xci, xvi, di
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_vertex_from_arrays_kernel!(
        ratio_vertices, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, xvi::NTuple{ND}, di::NTuple{ND, T}
    ) where {N, ND, T}
    w = ntuple(_ -> zero(T), Val(N))
    # Vertex position
    cell_vertex = ntuple(d -> xvi[d][I[d]], Val(ND))
    if ND == 2
        ni = size(first(phase_arrays))
        total_weight = zero(T)
        w_vals = ntuple(_ -> zero(T), Val(N))
        x_v, y_v = cell_vertex
        for offset₁ in -1:0, offset₂ in -1:0
            i_cell = I[1] + offset₁
            j_cell = I[2] + offset₂
            if 1 <= i_cell <= ni[1] && 1 <= j_cell <= ni[2]
                x_c = xci[1][i_cell]
                y_c = xci[2][j_cell]
                wx = 1.0 - abs(x_v - x_c) / di[1]
                wy = 1.0 - abs(y_v - y_c) / di[2]
                weight = wx * wy
                total_weight += weight
                for k in 1:N
                    phase_val = phase_arrays[k][i_cell, j_cell]
                    w_vals = Base.setindex(w_vals, w_vals[k] + weight * phase_val, k)
                end
            end
        end
        # Normalize and clamp only once at the end
        if total_weight > eps(T)
            w = w_vals ./ total_weight
        else
            w = ntuple(_ -> one(T) / N, Val(N))
        end
        w = clamp.(w, zero(T), one(T))
        # Clean up very small values
        w = ntuple(i -> w[i] < T(1e-5) ? zero(T) : w[i], Val(N))
        total_phases = sum(w)
        if total_phases > eps(T)
            w = w ./ total_phases
        else
            w = ntuple(_ -> one(T) / N, Val(N))
        end
    elseif ND == 3
        ni = size(first(phase_arrays))
        w_vals = ntuple(_ -> zero(T), Val(N))
        total_weight = zero(T)
        vertex_pos = ntuple(d -> xvi[d][I[d]], Val(ND))
        for offset₁ in -1:0, offset₂ in -1:0, offset₃ in -1:0
            i_cell = I[1] + offset₁
            j_cell = I[2] + offset₂
            k_cell = I[3] + offset₃
            if 1 <= i_cell <= ni[1] && 1 <= j_cell <= ni[2] && 1 <= k_cell <= ni[3]
            cell_center = (xci[1][i_cell], xci[2][j_cell], xci[3][k_cell])
            # Use trilinear weights for 3D interpolation
            weight = 1.0
            for d in 1:3
                weight *= 1.0 - abs(vertex_pos[d] - cell_center[d]) / di[d]
            end
            total_weight += weight
            for k in 1:N
                phase_val = phase_arrays[k][i_cell, j_cell, k_cell]
                w_vals = Base.setindex(w_vals, w_vals[k] + weight * phase_val, k)
            end
            end
        end
        # Normalize and clamp only once at the end
        if total_weight > eps(T)
            w = w_vals ./ total_weight
        else
            w = ntuple(_ -> one(T) / N, Val(N))
        end
        w = clamp.(w, zero(T), one(T))
        # Clean up very small values
        w = ntuple(i -> w[i] < T(1e-5) ? zero(T) : w[i], Val(N))
        total_phases = sum(w)
        if total_phases > eps(T)
            w = w ./ total_phases
        else
            w = ntuple(_ -> one(T) / N, Val(N))
        end
    end
    # Write to vertex grid
    for ip in cellaxes(ratio_vertices)
        @index ratio_vertices[ip, I...] = w[ip]
    end
    return nothing
end


function phase_ratios_face_from_arrays!(
        phase_face, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, dimension::Symbol
    ) where {N, ND}
    ni = size(first(phase_arrays))  # Cell grid size
    face_size = size(phase_face)  # Face grid size (including phase dimension)
    # ni_face = face_size[2:end]  # Face grid size excluding phase dimension
    di = compute_dx(xci)
    offsets = face_offset(Val(ND), dimension)

    @parallel (@idx ni) phase_ratios_face_from_arrays_kernel!(
        phase_face, phase_arrays, xci, di, offsets, ni
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_face_from_arrays_kernel!(
        ratio_faces, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, di::NTuple{ND, T}, offsets, ni
    ) where {N, ND, T}

    # Face index I corresponds to the face grid position
    w = ntuple(_ -> zero(T), Val(N))
    total_weight = zero(T)
    cell_center = getindex.(xci, I)
    cell_face = @. cell_center + di * offsets / 2
    ni = size(first(phase_arrays))
    # For staggered face grids, a face at position I lies between cells
    # We average from the two adjacent cells along the staggered dimension
    for side in 0:1
        # Calculate which cell this face point samples from
        cell_index = ntuple(d -> begin
            if offsets[d] == 1  # This dimension is staggered
                # Face I[d] is between cells I[d]-1 and I[d]
                # side=0 gives left cell (I[d]-1), side=1 gives right cell (I[d])
                I[d] - 1 + side
            else
                # Non-staggered dimension: use face index directly
                I[d]
            end
        end, Val(ND))

        # Check if cell index is within bounds
        valid_cell = all(1 ≤ cell_index[d] ≤ ni[d] for d in 1:ND)
        !valid_cell && continue

        # Equal weighting from both adjacent cells
        weight = T(0.5)
        total_weight += weight

        # Accumulate weighted phase values from this cell
        for k in 1:N
            phase_val = phase_arrays[k][cell_index...]
            w = Base.setindex(w, w[k] + weight * phase_val, k)
        end
    end

    # Normalize and clamp only once at the end
    if total_weight > eps(T)
        w = w ./ total_weight
    else
        w = ntuple(_ -> one(T) / N, Val(N))
    end
    w = clamp.(w, zero(T), one(T))
    # Clean up very small values (round to zero if < 1e-5)
    w = ntuple(i -> w[i] < T(1e-5) ? zero(T) : w[i], Val(N))
    total_phases = sum(w)
    if total_phases > eps(T)
        w = w ./ total_phases
    else
        w = ntuple(_ -> one(T) / N, Val(N))
    end

    for ip in cellaxes(ratio_faces)
        @index ratio_faces[ip, (I .+ offsets)...] = w[ip]
    end

    return nothing
end

## ============================================================================
## MIDPOINT VALUES - Analogous to phase_ratios_midpoint! (shear stress nodes)
## ============================================================================

function phase_ratios_midpoint_from_arrays!(
        phase_midpoints, phase_arrays::NTuple{N, AbstractArray}, xci, dimension
    ) where {N}
    ni = size(first(phase_arrays))  # Cell grid size
    midpoint_size = size(phase_midpoints)  # Midpoint grid size (including phase dimension)
    ni_midpoint = midpoint_size[2:end]  # Midpoint grid size excluding phase dimension
    di = compute_dx(xci)

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

    @parallel (@idx ni_midpoint) phase_ratios_midpoint_from_arrays_kernel!(
        phase_midpoints, phase_arrays, xci, di, offsets, ni
    )
    return nothing
end

@parallel_indices (I...) function phase_ratios_midpoint_from_arrays_kernel!(
        ratio_midpoints, phase_arrays::NTuple{N, AbstractArray}, xci::NTuple{ND}, di::NTuple{ND, T}, offsets, ni
    ) where {N, ND, T}

    # I represents the midpoint grid position
    w = ntuple(_ -> zero(T), Val(N))
    total_weight = zero(T)

    # For midpoint grids, we average from the corner cells
    # Number of corners depends on how many dimensions are staggered
    num_staggered = count(x -> x == 1, offsets)
    num_corners = 2^num_staggered

    # Iterate over all corner combinations
    for corner_bits in 0:(num_corners-1)
        # Calculate which cell this corner corresponds to
        bit_index = 0
        cell_index = ntuple(d -> begin
            if offsets[d] == 1  # This dimension is staggered
                # Extract the bit for this staggered dimension
                corner_bit = (corner_bits >> bit_index) & 1
                bit_index += 1
                # Midpoint I[d] is between cells I[d]-1 and I[d]
                # corner_bit=0 gives left cell (I[d]-1), corner_bit=1 gives right cell (I[d])
                I[d] - 1 + corner_bit
            else
                # Non-staggered dimension: use midpoint index directly
                I[d]
            end
        end, Val(ND))

        # Check if cell index is within bounds
        valid_cell = all(1 ≤ cell_index[d] ≤ ni[d] for d in 1:ND)
        !valid_cell && continue

        # Equal weighting from all corner cells
        weight = T(1.0) / num_corners
        total_weight += weight

        # Accumulate weighted phase values from this corner cell
        for k in 1:N
            phase_val = phase_arrays[k][cell_index...]
            w = Base.setindex(w, w[k] + weight * phase_val, k)
        end
    end

    # Normalize weights to ensure they sum to 1
    if total_weight > eps(T)
        w = w ./ total_weight
    else
        w = ntuple(_ -> one(T) / N, Val(N))
    end
    w = clamp.(w, zero(T), one(T))
    # Clean up very small values (round to zero if < 1e-5)
    w = ntuple(i -> w[i] < T(1e-5) ? zero(T) : w[i], Val(N))
    total_phases = sum(w)
    if total_phases > eps(T)
        w = w ./ total_phases
    else
        w = ntuple(_ -> one(T) / N, Val(N))
    end
    # Write to midpoint grid - use cellaxes like the original implementation
    for ip in cellaxes(ratio_midpoints)
        @index ratio_midpoints[ip, (I .+ offsets)...] = w[ip]
    end

    return nothing
end
