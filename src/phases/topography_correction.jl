import JustPIC._2D: cell_index, interp1D_inner, interp1D_extremas, distance
using StaticArrays

function update_phases_given_markerchain!(
        phase, chain::MarkerChain{backend}, particles::Particles{backend}, origin, di, air_phase
    ) where {backend}
    (; coords, index) = particles
    dy = di[2]
    return @parallel (1:size(index, 1)) _update_phases_given_markerchain!(
        phase, coords, index, chain.coords, chain.cell_vertices, origin, dy, air_phase
    )
end

@parallel_indices (icell) function _update_phases_given_markerchain!(
        phase, coords, index, chain_coords, cell_vertices, origin, dy, air_phase
    )
    _update_phases_given_markerchain_kernel!(
        phase, coords, index, chain_coords, cell_vertices, origin, dy, air_phase, icell
    )
    return nothing
end

function _update_phases_given_markerchain_kernel!(
        phase, coords, index, chain_coords, cell_vertices, origin, dy, air_phase, icell
    )
    T = eltype(eltype(phase))
    chain_yi = @cell chain_coords[2][icell]
    min_cell_j, max_cell_j = find_minmax_cell_indices(chain_yi, origin[2], dy)
    min_cell_j = max(1, min_cell_j - 10)
    max_cell_j = min(size(index, 2), max_cell_j + 10)
    cell_range = min_cell_j:max_cell_j

    # iterate over cells with marker chain on them
    for j in cell_range
        # iterate over particles j-th cell
        for ip in cellaxes(index)
            (@index index[ip, icell, j]) || continue
            xq = @index coords[1][ip, icell, j]
            yq = @index coords[2][ip, icell, j]
            phaseq = @index phase[ip, icell, j]

            # check if particle is above the marker chain
            above = is_above_chain(xq, yq, chain_coords, cell_vertices)
            # if the particle is above the surface and the phase is not air, set the phase to air
            if above && phaseq != air_phase
                @index phase[ip, icell, j] = T(air_phase)
                # @index coords[1][ip, icell, j] = NaN
                # @index coords[2][ip, icell, j] = NaN
                # @index index[ip, icell, j] = false
            end
            # if the particle is above the surface and the phase is air, set the phase to the closes rock phase
            if !above && phaseq == air_phase
                @index phase[ip, icell, j] = closest_phase(
                    coords, (xq, yq), index, ip, phase, air_phase, icell, j
                )
                # @index coords[1][ip, icell, j] = NaN
                # @index coords[2][ip, icell, j] = NaN
                # @index index[ip, icell, j] = false
            end
        end
    end

    return nothing
end

## Utils

function extrema_CA(x::AbstractArray)
    max_val = x[1]
    min_val = x[1]
    for i in 2:length(x)
        xᵢ = x[i]
        isnan(xᵢ) && continue
        if xᵢ > max_val
            max_val = xᵢ
        end
        if xᵢ < min_val
            min_val = xᵢ
        end
    end
    return min_val, max_val
end

function find_minmax_cell_indices(chain_yi, origin_y, dy)
    ymin, ymax = extrema_CA(chain_yi)
    min_cell_j = Int((ymin - origin_y) ÷ dy) + 1
    max_cell_j = Int((ymax - origin_y) ÷ dy) + 1
    return min_cell_j, max_cell_j
end

@inline function is_above_chain(xq, yq, coords, cell_vertices)
    I = cell_index(xq, cell_vertices)
    x_cell, y_cell = coords[1][I], coords[2][I]
    ychain = if 1 < I[1] < length(cell_vertices) - 1
        interp1D_inner(xq, x_cell, y_cell, coords, I)
    else
        interp1D_extremas(xq, x_cell, y_cell)
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
                d = distance(pxi, pn)
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
