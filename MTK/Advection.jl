abstract type AbstractParticles end

struct SemiLagrangianParticles{N,T} <: AbstractParticles
    px_i::NTuple{N,T}
    vpx_i::NTuple{N,T}
    Tpx_i::T

    function SemiLagrangianParticles(xvi::NTuple{2,T}) where {T}
        nx, ny = length.(xvi)
        n = nx*ny
        px_i = ntuple(Val(2)) do _
            @zeros(n)
        end
        for j in 1:ny, i in 1:nx
            px_i[1][linIdx(i, j, nx)] = xvi[1][i]
            px_i[2][linIdx(i, j, nx)] = xvi[2][j]
        end
        vpx_i = similar.(px_i)
        Tpx_i = similar(px_i[1])
        return new{2,typeof(Tpx_i)}(px_i, vpx_i, Tpx_i)
    end
end

@inline linIdx(i::T, j::T, nx::T) where {T<:Integer} = i + nx * (j - 1)

function advect_particles0!(
    p::SemiLagrangianParticles, xvi::NTuple{2,T}, Velocity::NTuple{2,F}, dt
) where {T,F}
    # unpack
    vx, vy = Velocity
    min_x, max_x = extrema(xvi[1])
    min_y, max_y = extrema(xvi[2])
    px, py = p.px_i
    nx, ny = length.(xvi)

    @parallel_indices (i, j) function advect_particles(px, py, xvi, vx, vy)
        @inbounds px[linIdx(i, j, nx)] = clamp(
            xvi[1][i] + dt * 0.5 * vx[i, j], min_x, max_x
        )
        @inbounds py[linIdx(i, j, nx)] = clamp(
            xvi[2][j] + dt * 0.5 * vy[i, j], min_y, max_y
        )
        return nothing
    end

    @parallel (1:nx, 1:ny) advect_particles(px, py, xvi, vx, vy)

    return nothing
end

function advect_particles!(p::SemiLagrangianParticles, xvi::NTuple{2,T}, dt) where {T}
    # unpack
    min_x, max_x = extrema(xvi[1])
    min_y, max_y = extrema(xvi[2])
    px, py = p.px_i
    vpx, vpy = p.vpx_i
    nx, ny = length.(xvi)

    @parallel_indices (i, j) function advect_particles(px, py, xvi, vx, vy)
        @inbounds px[linIdx(i, j, nx)] = clamp(
            xvi[1][i] + dt * 0.5 * vx[linIdx(i, j, nx)], min_x, max_x
        )
        @inbounds py[linIdx(i, j, nx)] = clamp(
            xvi[2][j] + dt * 0.5 * vy[linIdx(i, j, nx)], min_y, max_y
        )
        return nothing
    end

    @parallel (1:nx, 1:ny) advect_particles(px, py, xvi, vpx, vpy)

    return nothing
end

@parallel_indices (i) function advect_T!(T, Tpx_i)
    T[i] = Tpx_i[i]
    return nothing
end

function semilagrangian_advection_RK2!(T, parts_semilagrange, Velocity, xvi, dt)
    # 1st step advection 
    advect_particles0!(parts_semilagrange, xvi, Velocity, dt)

    # Interpolate to new px_i
    for (vpx_i, vx_i) in zip(parts_semilagrange.vpx_i, Velocity)
        grid2particle!(vpx_i, xvi, vx_i, parts_semilagrange.px_i)
    end

    # 2nd step advection 
    advect_particles!(parts_semilagrange, xvi, dt)

    # Interpolate T to back-tracked position
    grid2particle!(parts_semilagrange.Tpx_i, xvi, T, parts_semilagrange.px_i)

    # use backtracked T as new T in computational grid
    n = prod(length.(xvi))
    @parallel (1:n) advect_T!(T, parts_semilagrange.Tpx_i)

    return nothing
end

# function dike_tracers_advection!(T, Tracers, Velocity, xvi, dt; adv_scheme = :RK2)
#     if adv_scheme === :RK2
#         semilagrangian_advection_RK2!(T, parts_semilagrange, Velocity, xvi, dt)
#     end
# end

# function dike_tracers_advection_RK2!(T, Tracers, Velocity, xvi, dt)
#     # 1st step advection 
#     advect_particles0!(parts_semilagrange, xvi, Velocity, dt) 

#     # Interpolate to new px_i
#     for (vpx_i, vx_i) in zip(parts_semilagrange.vpx_i, Velocity)
#         grid2particle!(vpx_i, xvi, vx_i, parts_semilagrange.px_i)
#     end

#     # 2nd step advection 
#     advect_particles!(parts_semilagrange, xvi, dt) 

#     # Interpolate T to back-tracked position
#     grid2particle!(parts_semilagrange.Tpx_i, xvi, T, parts_semilagrange.px_i)

#     # use backtracked T as new T in computational grid
#     n = prod(length.(xvi))
#     @parallel (1:n) advect_T!(T, parts_semilagrange.Tpx_i)

#     return nothing
# end

# parts_semilagrange = SemiLagrangianParticles(xvi)

# ADVECTION OF DIKE PARTICLES
