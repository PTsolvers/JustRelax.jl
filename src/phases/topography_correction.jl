"""
    update_phases_given_markerchain!(phase, chain::MarkerChain, particles::Particles, origin, di, air_phase, args = ())

Deactivate the particles that end up on the wrong side of the free surface tracked by
`chain`: air-phase particles below it and rock particles above it. The topography is
linearly interpolated between `chain.cell_vertices` and `chain.h_vertices`. Their coordinates and
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
    return @parallel (1:(size(index, 1) - 2), 1:(size(index, 2) - 2)) _update_phases_given_markerchain!(
        phase,
        coords,
        index,
        chain.cell_vertices,
        chain.h_vertices,
        air_phase,
        args,
    )
end

@parallel_indices (icell, jcell) function _update_phases_given_markerchain!(
        phase, coords, index, cell_vertices, h_vertices, air_phase, args::NTuple{N, Any}
    ) where {N}
    particle_i, particle_j = icell + 1, jcell + 1
    for ip in cellaxes(index)
        (@index index[ip, particle_i, particle_j]) || continue

        xq = @index coords[1][ip, particle_i, particle_j]
        yq = @index coords[2][ip, particle_i, particle_j]
        phaseq = @index phase[ip, particle_i, particle_j]

        above = _is_above_chain(xq, yq, h_vertices, cell_vertices, icell)
        wrong_side = (above && phaseq != air_phase) || (!above && phaseq == air_phase)
        wrong_side || continue

        @index phase[ip, particle_i, particle_j] = NaN
        @index coords[1][ip, particle_i, particle_j] = NaN
        @index coords[2][ip, particle_i, particle_j] = NaN
        @index index[ip, particle_i, particle_j] = false

        for argᵢ in args
            @index argᵢ[ip, particle_i, particle_j] = NaN
        end
    end
    return nothing
end

@inline function _is_above_chain(xq, yq, h_vertices, cell_vertices, i::Integer)
    x0, x1 = cell_vertices[i], cell_vertices[i + 1]
    y0, y1 = h_vertices[i], h_vertices[i + 1]
    ychain = muladd((xq - x0) / (x1 - x0), y1 - y0, y0)
    return yq > ychain
end
