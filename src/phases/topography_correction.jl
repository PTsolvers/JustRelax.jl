using StaticArrays

"""
    update_phases_given_markerchain!(phase, chain::MarkerChain, particles::Particles, origin, di, air_phase, args = ())

Deactivate the particles that end up on the wrong side of the free surface tracked by
`chain`: air-phase particles below it and rock particles above it. Their coordinates and
every field in `args` are set to `NaN` and their index entry to `false`, so that particle
injection re-seeds those cells from their neighbours.

`origin` and `di` are the origin and grid spacing of the particle grid, and `air_phase`
the phase index standing for air.
"""
function update_phases_given_markerchain!(
        phase, chain::MarkerChain{backend}, particles::Particles{backend}, origin, di, air_phase
    ) where {backend}
    return update_phases_given_markerchain!(phase, chain, particles, origin, di, air_phase, ())
end

function update_phases_given_markerchain!(
        phase, chain::MarkerChain{backend}, particles::Particles{backend}, origin, di, air_phase, args::NTuple{N, Any}
    ) where {backend, N}
    (; coords, index) = particles
    return @parallel (1:size(chain.coords[1], 1)) _update_phases_given_markerchain!(
        phase,
        coords,
        index,
        chain.coords,
        chain.cell_vertices,
        origin,
        di,
        air_phase,
        args,
    )
end

@parallel_indices (icell) function _update_phases_given_markerchain!(
        phase, coords, index, chain_coords, cell_vertices, origin, di, air_phase, args::NTuple{N, Any}
    ) where {N}
    _update_phases_given_markerchain_kernel!(
        phase, coords, index, chain_coords, cell_vertices, origin, di, air_phase, icell, args
    )
    return nothing
end

function _update_phases_given_markerchain_kernel!(
        phase, coords, index, chain_coords, cell_vertices, origin, di, air_phase, icell, args::NTuple{N, Any}
    ) where {N}
    chain_yi = @cell chain_coords[2][icell]
    min_cell_j, max_cell_j = find_minmax_cell_indices(chain_yi, origin[2], di)
    # A chain column can temporarily contain no valid markers after advection or
    # resampling.  `extrema_CA` represents that case by an empty interval.
    if !isfinite(min_cell_j) || !isfinite(max_cell_j)
        return nothing
    end

    min_cell_j = max(1, min_cell_j - 10)
    max_cell_j = min(size(index, 2) - 2, max_cell_j + 10)
    # Do not construct a descending range: `a:b` with a > b still iterates in
    # Julia, and would address invalid particle rows for a chain wholly outside
    # the local particle domain.
    min_cell_j > max_cell_j && return nothing
    cell_range = min_cell_j:max_cell_j

    # iterate over cells with marker chain on them
    for j in cell_range
        particle_i, particle_j = icell + 1, j + 1
        # iterate over particles j-th cell
        for ip in cellaxes(index)
            (@index index[ip, particle_i, particle_j]) || continue
            xq = @index coords[1][ip, particle_i, particle_j]
            yq = @index coords[2][ip, particle_i, particle_j]
            phaseq = @index phase[ip, particle_i, particle_j]

            # check if particle is above the marker chain
            above = is_above_chain(xq, yq, chain_coords, cell_vertices)
            # if the particle is above the surface and the phase is not air, set the phase to air
            if above && phaseq != air_phase
                # @index phase[ip, icell, j] = T(air_phase)
                @index coords[1][ip, particle_i, particle_j] = NaN
                @index coords[2][ip, particle_i, particle_j] = NaN
                @index index[ip, particle_i, particle_j] = false

                for argᵢ in args
                    @index argᵢ[ip, particle_i, particle_j] = NaN
                end

            end
            # if the particle is above the surface and the phase is air, set the phase to the closes rock phase
            if !above && phaseq == air_phase
                # @index phase[ip, icell, j] = closest_phase(
                #     coords, (xq, yq), index, ip, phase, air_phase, icell, j
                # )
                @index coords[1][ip, particle_i, particle_j] = NaN
                @index coords[2][ip, particle_i, particle_j] = NaN
                @index index[ip, particle_i, particle_j] = false

                for argᵢ in args
                    @index argᵢ[ip, particle_i, particle_j] = NaN
                end
            end
        end
    end

    return nothing
end

## Utils

function extrema_CA(x::AbstractArray)
    max_val = -Inf
    min_val = Inf
    for i in eachindex(x)
        xᵢ = x[i]
        isfinite(xᵢ) || continue
        if xᵢ > max_val
            max_val = xᵢ
        end
        if xᵢ < min_val
            min_val = xᵢ
        end
    end
    return min_val, max_val
end

function find_minmax_cell_indices(chain_yi, origin_y, di)
    ymin, ymax = extrema_CA(chain_yi)
    isfinite(ymin) && isfinite(ymax) || return (1, 0)
    dy = @dy(di, 1)
    # `÷` truncates toward zero for negative floating-point quotients.  Cell
    # lookup needs mathematical floor so that coordinates below the origin are
    # assigned to the correct row.
    min_cell_j = Int(floor((ymin - origin_y) / dy)) + 1
    max_cell_j = Int(floor((ymax - origin_y) / dy)) + 1
    return min_cell_j, max_cell_j
end

# Uniform grid: row spacing is constant, so the row containing `y` is a single division.
@inline function locate_row_index(y, origin_y, dy::Number)
    return Int((y - origin_y) ÷ dy) + 1
end

# Non-uniform grid: `dy` holds the per-row cell heights (row 1 starting at `origin_y`), so no
# single spacing locates the row — walk the cumulative height. The loop must stay allocation
# free: this runs inside a `@parallel_indices` kernel and has to compile on CUDA/AMDGPU.
@inline function locate_row_index(y, origin_y, dy::AbstractVector)
    acc = origin_y
    for j in eachindex(dy)
        acc += dy[j]
        y <= acc && return j
    end
    return lastindex(dy)
end

@inline function is_above_chain(xq, yq, coords, cell_vertices)
    I = cell_index(xq, cell_vertices)
    x_cell, y_cell = coords[1][I], coords[2][I]
    ychain = if 1 < I[1] < length(cell_vertices) - 1
        JustPIC.interp1D_inner(xq, x_cell, y_cell, coords, I)
    else
        JustPIC.interp1D_extremas(xq, x_cell, y_cell)
    end
    return yq > ychain
end

# find closest phase different than the given `skip_phase`
function closest_phase(
        coords, pn, index, current_particle, phases, skip_phase, I::Vararg{Int, N}
    ) where {N}
    new_phase = @index phases[current_particle, I...]
    dist_min = Inf
    px, py = coords
    nx, ny = size(index)
    i, j = I
    for j in (j - 1):(j + 1)
        !(1 ≤ j ≤ ny) && continue
        for i in (i - 1):(i + 1)
            !(1 ≤ i ≤ nx) && continue

            for ip in cellaxes(index)
                # early escape conditions
                (ip == current_particle) && continue # current particle
                (@index index[ip, i, j]) || continue
                # get the phase of the particle and skip if it is the same as the `skip_phase`
                phaseᵢ = @index phases[ip, i, j]
                phaseᵢ == skip_phase && continue

                # distance from new point to the existing particle
                pxi = @index(px[ip, i, j]), @index(py[ip, i, j])
                d = JustPIC.distance(pxi, pn)
                # update the closest phase
                if d < dist_min
                    new_phase = phaseᵢ
                    dist_min = d
                end
            end
        end
    end

    return new_phase
end
