using StaticArrays

# Vorticity tensor

function compute_vorticity!(stokes::JustRelax.StokesArrays, _di, ni, ::Val{2})
    return @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.xy, @velocity(stokes)..., _di.velocity[1], _di.velocity[2]
    )
end

function compute_vorticity!(stokes::JustRelax.StokesArrays, _di, ni, ::Val{3})
    return @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.yz, stokes.ω.xz, stokes.ω.xy, @velocity(stokes)..., _di.velocity...
    )
end

@parallel_indices (I...) function compute_vorticity!(ωxy, Vx, Vy, _di_vx, _di_vy)

    Base.@propagate_inbounds @inline dx(A, I::Vararg{Int, 2}) = _d_xa(A, _dx, I...)
    Base.@propagate_inbounds @inline dy(A, I::Vararg{Int, 2}) = _d_ya(A, _dy, I...)

    i, j = I
    _dx = @dx(_di_vy, i)
    _dy = @dy(_di_vx, j)

    @inbounds ωxy[I...] = 0.5 * (dx(Vy, I...) - dy(Vx, I...))

    return nothing
end

@parallel_indices (I...) function compute_vorticity!(
        ωyz, ωxz, ωxy, Vx, Vy, Vz, _di
    )
    _dx, _dy, _dz = @dxi(_di, I...)
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

@parallel_indices (I...) function compute_vorticity!(
        ωyz, ωxz, ωxy, Vx, Vy, Vz, _di_vx, _di_vy, _di_vz
    )
    i, j, k = I

    if all(I .≤ size(ωyz))
        _dy_vz = @dy(_di_vz, j)
        _dz_vy = @dz(_di_vy, k)
        ∂Vz∂y = _dy_vz * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
        ∂Vy∂z = _dz_vy * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k])
        @inbounds ωyz[I...] = 0.5 * (∂Vz∂y - ∂Vy∂z)
    end
    if all(I .≤ size(ωxz))
        _dz_vx = @dz(_di_vx, k)
        _dx_vz = @dx(_di_vz, i)
        ∂Vx∂z = _dz_vx * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k])
        ∂Vz∂x = _dx_vz * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
        @inbounds ωxz[I...] = 0.5 * (∂Vx∂z - ∂Vz∂x)
    end
    if all(I .≤ size(ωxy))
        _dx_vy = @dx(_di_vy, i)
        _dy_vx = @dy(_di_vx, j)
        ∂Vy∂x = _dx_vy * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
        ∂Vx∂y = _dy_vx * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1])
        @inbounds ωxy[I...] = 0.5 * (∂Vy∂x - ∂Vx∂y)
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

        τ_rotated = GeoParams.rotate_elastic_stress2D(ω_xy, (τ_xx, τ_yy, τ_xy), dt)

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

        components = xx, yy, zz, yz, xz, xy
        Base.@nexprs 6 i -> begin
            @inline @inbounds @index components[i][ip, I...] = τ_rotated[i]
        end
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

        tmp = τ_xy * ω_xy * 2
        @inbounds @index xx[ip, I...] = muladd(dt, tmp, τ_xx)
        @inbounds @index yy[ip, I...] = muladd(dt, tmp, τ_yy)
        @inbounds @index xy[ip, I...] = muladd(dt, (τ_xx - τ_yy) * ω_xy, τ_xy)
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
        stokes, τ_particles::JustRelax.StressParticles{backend}, particles
    ) where {backend}
    return stress2grid!(
        stokes,
        normal_stress(τ_particles)...,
        shear_stress(τ_particles)...,
        particles,
    )
end

function stress2grid!(stokes, pτxx, pτyy, pτxy, particles)
    # normal components
    particle2centroid!(stokes.τ_o.xx, pτxx, particles)
    particle2centroid!(stokes.τ_o.yy, pτyy, particles)
    particle2centroid!(stokes.τ_o.xy_c, pτxy, particles)
    # shear components
    particle2grid!(stokes.τ_o.xx_v, pτxx, particles)
    particle2grid!(stokes.τ_o.yy_v, pτyy, particles)
    particle2grid!(stokes.τ_o.xy, pτxy, particles)

    return nothing
end

function stress2grid!(stokes, pτxx, pτyy, pτzz, pτyz, pτxz, pτxy, particles)
    # normal components
    particle2centroid!(stokes.τ_o.xx, pτxx, particles)
    particle2centroid!(stokes.τ_o.yy, pτyy, particles)
    particle2centroid!(stokes.τ_o.zz, pτzz, particles)
    # shear components
    particle2grid!(stokes.τ_o.yz, pτyz, particles)
    particle2grid!(stokes.τ_o.xz, pτxz, particles)
    particle2grid!(stokes.τ_o.xy, pτxy, particles)

    return nothing
end

function rotate_stress!(
        τ_particles::JustRelax.StressParticles{backend}, stokes, particles, dt
    ) where {backend}
    return rotate_stress!(unwrap(τ_particles)..., stokes, particles, dt)
end

function rotate_stress!(pτxx, pτyy, pτxy, pω, stokes, particles, dt)
    # normal components
    centroid2particle!(pτxx, stokes.τ.xx, particles)
    centroid2particle!(pτyy, stokes.τ.yy, particles)
    # shear components
    grid2particle!(pτxy, stokes.τ.xy, particles)
    # vorticity tensor
    grid2particle!(pω, stokes.ω.xy, particles)
    # rotate stress
    rotate_stress_particles!((pτxx, pτyy, pτxy), (pω,), particles, dt)

    return nothing
end

function rotate_stress!(
        pτxx, pτyy, pτzz, pτyz, pτxz, pτxy, pωyz, pωxz, pωxy, stokes, particles, dt
    )
    # normal components
    centroid2particle!(pτxx, stokes.τ.xx, particles)
    centroid2particle!(pτyy, stokes.τ.yy, particles)
    centroid2particle!(pτzz, stokes.τ.zz, particles)
    # shear components
    grid2particle!(pτyz, stokes.τ.yz, particles)
    grid2particle!(pτxz, stokes.τ.xz, particles)
    grid2particle!(pτxy, stokes.τ.xy, particles)
    # vorticity tensor
    grid2particle!(pωyz, stokes.ω.yz, particles)
    grid2particle!(pωxz, stokes.ω.xz, particles)
    grid2particle!(pωxy, stokes.ω.xy, particles)
    # rotate stress
    rotate_stress_particles!(
        (pτxx, pτyy, pτzz, pτyz, pτxz, pτxy), (pωyz, pωxz, pωxy), particles, dt
    )

    return nothing
end
