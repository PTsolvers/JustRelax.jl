using StaticArrays

# ROTATION KERNELS

function jaumann!(xx, yy, xy, ω, index, dt)
    ni = size(xx)
    @parallel (@idx ni) _jaumann!(xx, yy, xy, ω, index, dt)
    return nothing
end

@parallel_indices (i, j) function _jaumann!(xx, yy, xy, ω, index, dt)
    cell = i, j

    for ip in JustRelax.cellaxes(index)
        !@cell(index[ip, cell...]) && continue # no particle in this location

        ω_xy = @cell ω[ip, cell...]
        τ_xx = @cell xx[ip, cell...]
        τ_yy = @cell yy[ip, cell...]
        τ_xy = @cell xy[ip, cell...]

        tmp = τ_xy * ω_xy * 2.0
        @cell xx[ip, cell...] = muladd(dt, tmp, τ_xx)
        @cell yy[ip, cell...] = muladd(dt, tmp, τ_yy)
        @cell xy[ip, cell...] = muladd(dt, (τ_xx - τ_yy) * ω_xy, τ_xy)
    end

    return nothing
end

function rotation_matrix!(xx, yy, xy, ω, index, dt)
    ni = size(xx)
    @parallel (@idx ni) _rotation_matrix!(xx, yy, xy, ω, index, dt)
    return nothing
end

@parallel_indices (i, j) function _rotation_matrix!(xx, yy, xy, ω, index, dt)
    cell = i, j

    for ip in JustRelax.cellaxes(index)
        !@cell(index[ip, cell...]) && continue # no particle in this location

        θ = dt * @cell ω[ip, cell...]
        sinθ, cosθ = sincos(θ)

        τ_xx = @cell xx[ip, cell...]
        τ_yy = @cell yy[ip, cell...]
        τ_xy = @cell xy[ip, cell...]

        R = @SMatrix [
            cosθ -sinθ
            sinθ cosθ
        ]

        τ = @SMatrix [
            τ_xx τ_xy
            τ_xy τ_yy
        ]

        # this could be fully unrolled in 2D
        τr = R * τ * R'

        @cell xx[ip, cell...] = τr[1, 1]
        @cell yy[ip, cell...] = τr[2, 2]
        @cell xy[ip, cell...] = τr[1, 2]
    end

    return nothing
end

# STRESS ROTATION ON THE PARTICLES

function rotate_stress_particles!(
    stokes::JustRelax.StokesArrays,
    τxx_vertex::T,
    τyy_vertex::T,
    τxx_o_vertex::T,
    τyy_o_vertex::T,
    τxx_p::CA,
    τyy_p::CA,
    τxy_p::CA,
    vorticity_p::CA,
    particles,
    grid::JustRelax.Geometry,
    dt;
    fn=rotation_matrix!,
) where {T<:AbstractArray,CA}
    (; xvi, xci) = grid
    nx, ny = size(τxx_p)

    # interpolate stresses to particles
    for (src, src_o, dst_v, dst_v_o, dst_p) in zip(
        @tensor_center(stokes.τ),
        @tensor_center(stokes.τ_o),
        (τxx_vertex, τyy_vertex, stokes.τ.xy),
        (τxx_o_vertex, τyy_o_vertex, stokes.τ_o.xy),
        (τxx_p, τyy_p, τxy_p),
    )
        @parallel center2vertex!(dst_v, src)
        @parallel center2vertex!(dst_v_o, src_o)
        @parallel (1:(nx + 1)) free_slip_y!(dst_v)
        @parallel (1:(ny + 1)) free_slip_x!(dst_v)
        @parallel (1:(nx + 1)) free_slip_y!(dst_v_o)
        @parallel (1:(ny + 1)) free_slip_x!(dst_v_o)
        grid2particle_flip!(dst_p, xvi, dst_v, dst_v_o, particles)
    end

    # interpolate vorticity to particles    
    @parallel center2vertex!(stokes.ω.xy_v, stokes.ω.xy_c)
    @parallel center2vertex!(stokes.ω_o.xy_v, stokes.ω_o.xy_c)
    @parallel (1:nx) free_slip_y!(stokes.ω.xy_c)
    @parallel (1:ny) free_slip_x!(stokes.ω.xy_c)
    @parallel (1:(nx + 1)) free_slip_y!(stokes.ω_o.xy_v)
    @parallel (1:(ny + 1)) free_slip_x!(stokes.ω_o.xy_v)
    grid2particle_flip!(vorticity_p, xvi, stokes.ω.xy_v, stokes.ω_o.xy_v, particles)
    # rotate old stress
    fn(τxx_p, τyy_p, τxy_p, vorticity_p, particles.index, dt)
    # interpolate old stress to grid arrays
    particle2grid_centroid!(stokes.τ_o.xx, τxx_p, xci, particles)
    particle2grid_centroid!(stokes.τ_o.yy, τyy_p, xci, particles)
    return particle2grid_centroid!(stokes.τ_o.xy_c, τxy_p, xci, particles)
end
