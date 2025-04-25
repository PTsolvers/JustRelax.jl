using StaticArrays

# Vorticity tensor

@parallel_indices (I...) function compute_vorticity!(ωxy, Vx, Vy, _dx, _dy)
    Base.@propagate_inbounds @inline dx(A) = _d_xa(A, _dx, I...)
    Base.@propagate_inbounds @inline dy(A) = _d_ya(A, _dy, I...)

    @inbounds ωxy[I...] = 0.5 * (dx(Vy) - dy(Vx))

    return nothing
end

@parallel_indices (I...) function compute_vorticity!(
        ωyz, ωxz, ωxy, Vx, Vy, Vz, _dx, _dy, _dz
    )
    Base.@propagate_inbounds @inline dx(A) = _d_xa(A, _dx, I...)
    Base.@propagate_inbounds @inline dy(A) = _d_ya(A, _dy, I...)
    Base.@propagate_inbounds @inline dz(A) = _d_za(A, _dz, I...)

    if all(I .≤ size(ωyz))
        @inbounds ωyz[I...] = 0.5 * (dy(Vz) - dz(Vy))
    end
    if all(I .≤ size(ωxz))
        @inbounds ωxz[I...] = 0.5 * (dz(Vx) - dx(Vz))
    end
    if all(I .≤ size(ωxy))
        @inbounds ωxy[I...] = 0.5 * (dx(Vy) - dy(Vx))
    end

    return nothing
end

## Stress Rotation on the particles

function rotate_stress_particles!(
        τ::NTuple, ω::NTuple, particles::Particles, dt; method::Symbol = :matrix
    )
    @parallel (@idx size(particles.index)) rotate_stress_particles_GeoParams!(
        τ..., ω..., particles.index, dt
    )
    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_GeoParams!(
        xx, yy, xy, ω, index, dt
    )
    for ip in cellaxes(index)
        @index(index[ip, I...]) || continue # no particle in this location

        ω_xy = @inbounds @index ω[ip, I...]
        τ_xx = @inbounds @index xx[ip, I...]
        τ_yy = @inbounds @index yy[ip, I...]
        τ_xy = @inbounds @index xy[ip, I...]

        τ_rotated = rotate_elastic_stress2D(ω_xy, (τ_xx, τ_yy, τ_xy), dt)

        @inbounds @index xx[ip, I...] = τ_rotated[1]
        @inbounds @index yy[ip, I...] = τ_rotated[2]
        @inbounds @index xy[ip, I...] = τ_rotated[3]
    end

    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_GeoParams!(
        xx, yy, zz, yz, xz, xy, ωyz, ωxz, ωxy, index, dt
    )
    for ip in cellaxes(index)
        @index(index[ip, I...]) || continue # no particle in this location

        ω_yz = @inbounds @index ωyz[ip, I...]
        ω_xz = @inbounds @index ωxz[ip, I...]
        ω_xy = @inbounds @index ωxy[ip, I...]
        τ_xx = @inbounds @index xx[ip, I...]
        τ_yy = @inbounds @index yy[ip, I...]
        τ_yz = @inbounds @index yz[ip, I...]
        τ_xz = @inbounds @index xz[ip, I...]
        τ_xy = @inbounds @index xy[ip, I...]

        τ_rotated = GeoParams.rotate_elastic_stress3D(
            (ω_yz, ω_xz, ω_xy), (τ_xx, τ_yy, τ_xy, τ_yz, τ_xz, τ_xy), dt
        )

        @inbounds @index xx[ip, I...] = τ_rotated[1]
        @inbounds @index yy[ip, I...] = τ_rotated[2]
        @inbounds @index zz[ip, I...] = τ_rotated[3]
        @inbounds @index yz[ip, I...] = τ_rotated[4]
        @inbounds @index xz[ip, I...] = τ_rotated[5]
        @inbounds @index xy[ip, I...] = τ_rotated[6]
    end

    return nothing
end

@parallel_indices (I) function rotate_stress_particles_jaumann!(xx, yy, xy, ω, index, dt)
    for ip in cellaxes(index)
        !@index(index[ip, I...]) && continue # no particle in this location

        ω_xy = @inbounds @index ω[ip, I...]
        τ_xx = @inbounds @index xx[ip, I...]
        τ_yy = @inbounds @index yy[ip, I...]
        τ_xy = @inbounds @index xy[ip, I...]

        tmp = τ_xy * ω_xy * 2.0
        @inbounds @index xx[ip, I...] = fma(dt, cte, τ_xx)
        @inbounds @index yy[ip, I...] = fma(dt, cte, τ_yy)
        @inbounds @index xy[ip, I...] = fma(dt, (τ_xx - τ_yy) * ω_xy, τ_xy)
    end

    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_rotation_matrix!(
        xx, yy, xy, ω, index, dt
    )
    for ip in cellaxes(index)
        !@index(index[ip, I...]) && continue # no particle in this location

        θ = @inbounds dt * @index ω[ip, I...]
        sinθ, cosθ = sincos(θ)

        τ_xx = @inbounds @index xx[ip, I...]
        τ_yy = @inbounds @index yy[ip, I...]
        τ_xy = @inbounds @index xy[ip, I...]

        R = @SMatrix [
            cosθ -sinθ
            sinθ cosθ
        ]

        τ = @SMatrix [
            τ_xx τ_xy
            τ_xy τ_yy
        ]

        # this could be fully unrolled in 2D
        τr = R * (τ * R')

        @inbounds @index xx[ip, I...] = τr[1, 1]
        @inbounds @index yy[ip, I...] = τr[2, 2]
        @inbounds @index xy[ip, I...] = τr[1, 2]
    end

    return nothing
end

# Interpolations between stress on the particles and the grid

function stress2grid!(
        stokes, τ_particles::JustRelax.StressParticles{backend}, xvi, xci, particles
    ) where {backend}
    return stress2grid!(
        stokes,
        normal_stress(τ_particles)...,
        shear_stress(τ_particles)...,
        xvi,
        xci,
        particles,
    )
end

function stress2grid!(stokes, pτxx, pτyy, pτxy, xvi, xci, particles)
    # normal components
    particle2centroid!(stokes.τ_o.xx, pτxx, xci, particles)
    particle2centroid!(stokes.τ_o.yy, pτyy, xci, particles)
    # shear components
    particle2grid!(stokes.τ_o.xy, pτxy, xvi, particles)

    return nothing
end

function stress2grid!(stokes, pτxx, pτyy, pτzz, pτyz, pτxz, pτxy, xvi, xci, particles)
    # normal components
    particle2centroid!(stokes.τ_o.xx, pτxx, xci, particles)
    particle2centroid!(stokes.τ_o.yy, pτyy, xci, particles)
    particle2centroid!(stokes.τ_o.zz, pτzz, xci, particles)
    # shear components
    particle2grid!(stokes.τ_o.yz, pτyz, xvi, particles)
    particle2grid!(stokes.τ_o.xz, pτxz, xvi, particles)
    particle2grid!(stokes.τ_o.xy, pτxy, xvi, particles)

    return nothing
end

function rotate_stress!(
        τ_particles::JustRelax.StressParticles{backend}, stokes, particles, xci, xvi, dt
    ) where {backend}
    return rotate_stress!(unwrap(τ_particles)..., stokes, particles, xci, xvi, dt)
end

function rotate_stress!(pτxx, pτyy, pτxy, pω, stokes, particles, xci, xvi, dt)
    # normal components
    centroid2particle!(pτxx, xci, stokes.τ.xx, particles)
    centroid2particle!(pτyy, xci, stokes.τ.yy, particles)
    # shear components
    grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
    # vorticity tensor
    grid2particle!(pω, xvi, stokes.ω.xy, particles)
    # rotate stress
    rotate_stress_particles!((pτxx, pτyy, pτxy), (pω,), particles, dt)

    return nothing
end

function rotate_stress!(
        pτxx, pτyy, pτzz, pτyz, pτxz, pτxy, pωyz, pωxz, pωxy, stokes, particles, xci, xvi, dt
    )
    # normal components
    centroid2particle!(pτxx, xci, stokes.τ.xx, particles)
    centroid2particle!(pτyy, xci, stokes.τ.yy, particles)
    centroid2particle!(pτzz, xci, stokes.τ.zz, particles)
    # shear components
    grid2particle!(pτyz, xvi, stokes.τ.yz, particles)
    grid2particle!(pτxz, xvi, stokes.τ.xz, particles)
    grid2particle!(pτxy, xvi, stokes.τ.xy, particles)
    # vorticity tensor
    grid2particle!(pωyz, xvi, stokes.ω.yz, particles)
    grid2particle!(pωxz, xvi, stokes.ω.xz, particles)
    grid2particle!(pωxy, xvi, stokes.ω.xy, particles)
    # rotate stress
    rotate_stress_particles!(
        (pτxx, pτyy, pτzz, pτyz, pτxz, pτxy), (pωyz, pωxz, pωxy), particles, dt
    )

    return nothing
end
