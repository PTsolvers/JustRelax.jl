# Velocity helper grids for the particle advection
function velocity_grids(xci, xvi, di::NTuple{2, T}) where T
    dx, dy  = di
    yVx     = LinRange(xci[2][1] - dy, xci[2][end] + dy, length(xci[2])+2)
    xVy     = LinRange(xci[1][1] - dx, xci[1][end] + dx, length(xci[1])+2)
    grid_vx = xvi[1], yVx
    grid_vy = xVy, xvi[2]

    return grid_vx, grid_vy
end

function velocity_grids(xci, xvi, di::NTuple{3, T}) where T
    xghost  = ntuple(Val(3)) do i
        LinRange(xci[i][1] - di[i], xci[i][end] + di[i], length(xci[i])+2)
    end
    grid_vx = xvi[1]   , xghost[2], xghost[3]
    grid_vy = xghost[1], xvi[2]   , xghost[3]
    grid_vz = xghost[1], xghost[2], xvi[3]

    return grid_vx, grid_vy, grid_vz
end

@inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...) 
@inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
@inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
@inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

function init_particles(nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, ni::NTuple{3, Int})
    ncells     = prod(ni)
    np         = max_xcell * ncells
    px, py, pz = ntuple(_ -> @fill(NaN, ni..., celldims=(max_xcell,)) , Val(3))
    inject     = @fill(false, ni..., eltype=Bool)
    index      = @fill(false, ni..., celldims=(max_xcell,), eltype=Bool) 
    
    @parallel_indices (i, j, k) function fill_coords_index(px, py, pz, index)
        @inline r()= rand(0.05:1e-5:0.95)
        I          = i, j, k
        # lower-left corner of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            @cell px[l, I...]    = x0 + dx * r()
            @cell py[l, I...]    = y0 + dy * r()
            @cell pz[l, I...]    = z0 + dz * r()
            @cell index[l, I...] = true
        end
        return nothing
    end

    @parallel (@idx ni) fill_coords_index(px, py, pz, index)    

    return Particles(
        (px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np, ni
    )
end