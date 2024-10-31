using StaticArrays

# Vorticity tensor

@parallel_indices (I...) function compute_vorticity!(ωxy, Vx, Vy, _dx, _dy)
    dx(A) = _d_xa(A, I..., _dx)
    dy(A) = _d_ya(A, I..., _dy)

    ωxy[I...] = 0.5 * (dx(Vy) - dy(Vx))

    return nothing
end

@parallel_indices (I...) function compute_vorticity!(
    ωyz, ωxz, ωxy, Vx, Vy, Vz, _dx, _dy, _dz
)
    dx(A) = _d_xa(A, I..., _dx)
    dy(A) = _d_ya(A, I..., _dy)
    dz(A) = _d_za(A, I..., _dz)

    if all(I .≤ size(ωyz))
        ωyz[I...] = 0.5 * (dy(Vz) - dz(Vy))
    end
    if all(I .≤ size(ωxz))
        ωxz[I...] = 0.5 * (dz(Vx) - dx(Vz))
    end
    if all(I .≤ size(ωxy))
        ωxy[I...] = 0.5 * (dx(Vy) - dy(Vx))
    end

    return nothing
end

## Stress Rotation on the particles

function rotate_stress_particles!(
    τ::NTuple, ω::NTuple, particles::Particles, dt; method::Symbol=:matrix
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

        ω_xy = @index ω[ip, I...]
        τ_xx = @index xx[ip, I...]
        τ_yy = @index yy[ip, I...]
        τ_xy = @index xy[ip, I...]

        τ_rotated = GeoParams.rotate_elastic_stress2D(ω_xy, (τ_xx, τ_yy, τ_xy), dt)

        @index xx[ip, I...] = τ_rotated[1]
        @index yy[ip, I...] = τ_rotated[2]
        @index xy[ip, I...] = τ_rotated[3]
    end

    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_GeoParams!(
    xx, yy, zz, yz, xz, xy, ωyz, ωxz, ωxy, index, dt
)
    for ip in cellaxes(index)
        @index(index[ip, I...]) || continue # no particle in this location

        ω_yz = @index ωyz[ip, I...]
        ω_xz = @index ωxz[ip, I...]
        ω_xy = @index ωxy[ip, I...]
        τ_xx = @index xx[ip, I...]
        τ_yy = @index yy[ip, I...]
        τ_yz = @index yz[ip, I...]
        τ_xz = @index xz[ip, I...]
        τ_xy = @index xy[ip, I...]

        τ_rotated = GeoParams.rotate_elastic_stress3D(
            (ω_yz, ω_xz, ω_xy), (τ_xx, τ_yy, τ_xy, τ_yz, τ_xz, τ_xy), dt
        )

        @index xx[ip, I...] = τ_rotated[1]
        @index yy[ip, I...] = τ_rotated[2]
        @index zz[ip, I...] = τ_rotated[3]
        @index yz[ip, I...] = τ_rotated[4]
        @index xz[ip, I...] = τ_rotated[5]
        @index xy[ip, I...] = τ_rotated[6]
    end

    return nothing
end

@parallel_indices (I) function rotate_stress_particles_jaumann!(xx, yy, xy, ω, index, dt)
    for ip in cellaxes(index)
        !@index(index[ip, I...]) && continue # no particle in this location

        ω_xy = @index ω[ip, I...]
        τ_xx = @index xx[ip, I...]
        τ_yy = @index yy[ip, I...]
        τ_xy = @index xy[ip, I...]

        tmp = τ_xy * ω_xy * 2.0
        @index xx[ip, I...] = fma(dt, cte, τ_xx)
        @index yy[ip, I...] = fma(dt, cte, τ_yy)
        @index xy[ip, I...] = fma(dt, (τ_xx - τ_yy) * ω_xy, τ_xy)
    end

    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_rotation_matrix!(
    xx, yy, xy, ω, index, dt
)
    for ip in cellaxes(index)
        !@index(index[ip, I...]) && continue # no particle in this location

        θ = dt * @index ω[ip, I...]
        sinθ, cosθ = sincos(θ)

        τ_xx = @index xx[ip, I...]
        τ_yy = @index yy[ip, I...]
        τ_xy = @index xy[ip, I...]

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

        @index xx[ip, I...] = τr[1, 1]
        @index yy[ip, I...] = τr[2, 2]
        @index xy[ip, I...] = τr[1, 2]
    end

    return nothing
end

## Stress Rotation on the grid

@parallel_indices (I...) function rotate_stress!(V, τ::NTuple{3,T}, _di, dt) where {T}
    @inbounds rotate_stress!(V, τ, tuple(I...), _di, dt)
    return nothing
end

@inline function tensor2voigt(xx, yy, xy, i, j)
    av(A) = _av_a(A, i, j)

    voigt = xx[i, j], yy[i, j], av(xy)

    return voigt
end

@inline function tensor2voigt(xx, yy, zz, yz, xz, xy, i, j, k)
    av_yz(A) = _av_yz(A, i, j, k)
    av_xz(A) = _av_xz(A, i, j, k)
    av_xy(A) = _av_xy(A, i, j, k)

    voigt = xx[i, j], yy[i, j], zz[i, j], av_yz(yz), av_xz(xz), av_xy(xy)

    return voigt
end

"""
    Jaumann derivative

τij_o += v_k * ∂τij_o/∂x_k - ω_ij * ∂τkj_o + ∂τkj_o * ω_ij

"""
Base.@propagate_inbounds function rotate_stress!(
    V, τ::NTuple{N,T}, idx, _di, dt
) where {N,T}
    ## 1) Advect stress
    Vᵢⱼ = velocity2center(V..., idx...) # averages @ cell center
    τij_adv = advect_stress(τ..., Vᵢⱼ..., idx..., _di...)

    ## 2) Rotate stress
    # average ∂Vx/∂y @ cell center
    ∂V∂x = cross_derivatives(V..., _di..., idx...)
    # compute xy component of the vorticity tensor; normal components = 0.0
    ω = compute_vorticity(∂V∂x)
    # stress tensor in Voigt notation
    # τ_voigt = ntuple(Val(N)) do k
    #     Base.@_inline_meta
    #     τ[k][idx...]
    # end
    τ_voigt = tensor2voigt(τ..., idx...)

    # actually rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ω, τ_voigt, dt)

    R = @SMatrix [
        0.0 -ω
        ω 0.0
    ]

    τm = @SMatrix [
        τ_voigt[1] τ_voigt[3]
        τ_voigt[3] τ_voigt[2]
    ]
    τr = τm * R - R * τm
    τr_voigt = τr[1, 1], τr[2, 2], τr[1, 2]

    ## 3) Update stress
    for k in 1:N
        τ[k][idx...] = fma(τij_adv[k] * 0, dt, τr_voigt[k])
    end
    return nothing
end

# 2D
Base.@propagate_inbounds function advect_stress(τxx, τyy, τxy, Vx, Vy, i, j, _dx, _dy)
    τ = τxx, τyy, τxy
    τ_adv = ntuple(Val(3)) do k
        Base.@_inline_meta
        dx_right, dx_left, dy_up, dy_down = upwind_derivatives(τ[k], i, j)
        return advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    end
    return τ_adv
end

# 3D
Base.@propagate_inbounds function advect_stress(
    τxx, τyy, τzz, τyz, τxz, τxy, Vx, Vy, Vz, i, j, k, _dx, _dy, _dz
)
    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_adv = ntuple(Val(6)) do l
        Base.@_inline_meta
        dx_right, dx_left, dy_back, dy_front, dz_up, dz_down = upwind_derivatives(
            τ[l], i, j, k
        )
        return advection_term(
            Vx,
            Vy,
            Vz,
            dx_right,
            dx_left,
            dy_back,
            dy_front,
            dz_up,
            dz_down,
            _dx,
            _dy,
            _dz,
        )
    end
    return τ_adv
end

# 2D
Base.@propagate_inbounds function upwind_derivatives(A, i, j)
    nx, ny = size(A)
    center = A[i, j]
    # dx derivatives
    x_left = i - 1 > 1 ? A[i - 1, j] : 0.0
    x_right = i + 1 < nx ? A[i + 1, j] : 0.0
    dx_right = x_right - center
    dx_left = center - x_left
    # dy derivatives
    y_down = j - 1 > 1 ? A[i, j - 1] : 0.0
    y_up = j + 1 < ny ? A[i, j + 1] : 0.0
    dy_up = y_up - center
    dy_down = center - y_down

    return dx_right, dx_left, dy_up, dy_down
end

# 3D
Base.@propagate_inbounds function upwind_derivatives(A, i, j, k)
    nx, ny, nz = size(A)
    center = A[i, j, k]
    x_left = x_right = y_front = y_back = z_down = z_up = 0.0
    # dx derivatives
    i - 1 > 1 && (x_left = A[i - 1, j, k])
    i + 1 < nx && (x_right = A[i + 1, j, k])
    dx_right = x_right - center
    dx_left = center - x_left
    # dy derivatives
    j - 1 > 1 && (y_front = A[i, j - 1, k])
    j + 1 < ny && (y_back = A[i, j + 1, k])
    dy_back = y_back - center
    dy_front = center - y_front
    # dz derivatives
    k - 1 > 1 && (z_down = A[i, j, k - 1])
    k + 1 < nz && (z_up = A[i, j, k + 1])
    dz_up = z_up - center
    dz_down = center - z_down

    return dx_right, dx_left, dy_back, dy_front, dz_up, dz_down
end

# 2D
@inline function advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    return (Vx > 0 ? dx_right : dx_left) * Vx * _dx + (Vy > 0 ? dy_up : dy_down) * Vy * _dy
end

# 3D
@inline function advection_term(
    Vx, Vy, Vz, dx_right, dx_left, dy_back, dy_front, dz_up, dz_down, _dx, _dy, _dz
)
    return (Vx > 0 ? dx_right : dx_left) * Vx * _dx +
           (Vy > 0 ? dy_back : dy_front) * Vy * _dy +
           (Vz > 0 ? dz_up : dz_down) * Vz * _dz
end

# averages @ cell center 2D
Base.@propagate_inbounds function velocity2center(Vx, Vy, i, j)
    i1, j1 = @add 1 i j
    Vxᵢⱼ = 0.5 * (Vx[i, j1] + Vx[i1, j1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j] + Vy[i1, j1])
    return Vxᵢⱼ, Vyᵢⱼ
end

# averages @ cell center 3D
Base.@propagate_inbounds function velocity2center(Vx, Vy, Vz, i, j, k)
    i1, j1, k1 = @add 1 i j k
    Vxᵢⱼ = 0.5 * (Vx[i, j1, k1] + Vx[i1, j1, k1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j, k1] + Vy[i1, j1, k1])
    Vzᵢⱼ = 0.5 * (Vz[i1, j1, k] + Vz[i1, j1, k1])
    return Vxᵢⱼ, Vyᵢⱼ, Vzᵢⱼ
end

# 2D
Base.@propagate_inbounds function cross_derivatives(Vx, Vy, _dx, _dy, i, j)
    i1, j1 = @add 1 i j
    i2, j2 = @add 2 i j
    # average @ cell center
    ∂Vx∂y =
        0.25 *
        _dy *
        (
            Vx[i, j1] - Vx[i, j] + Vx[i, j2] - Vx[i, j1] + Vx[i1, j1] - Vx[i1, j] +
            Vx[i1, j2] - Vx[i1, j1]
        )
    ∂Vy∂x =
        0.25 *
        _dx *
        (
            Vy[i1, j] - Vy[i, j] + Vy[i2, j] - Vy[i1, j] + Vy[i1, j1] - Vy[i, j1] +
            Vy[i2, j1] - Vy[i1, j1]
        )
    return ∂Vx∂y, ∂Vy∂x
end

Base.@propagate_inbounds function cross_derivatives(Vx, Vy, Vz, _dx, _dy, _dz, i, j, k)
    i1, j1, k2 = @add 1 i j k
    i2, j2, k2 = @add 2 i j k
    # cross derivatives @ cell centers
    ∂Vx∂y =
        0.25 *
        _dy *
        (
            Vx[i, j1, k1] - Vx[i, j, k1] + Vx[i, j2, k1] - Vx[i, j1, k1] + Vx[i1, j1, k1] -
            Vx[i1, j, k1] + Vx[i1, j2, k1] - Vx[i1, j1, k1]
        )
    ∂Vx∂z =
        0.25 *
        _dz *
        (
            Vx[i, j1, k1] - Vx[i, j, k] + Vx[i, j2, k2] - Vx[i, j1, k1] + Vx[i1, j1, k1] -
            Vx[i1, j, k] + Vx[i1, j2, k2] - Vx[i1, j1, k1]
        )
    ∂Vy∂x =
        0.25 *
        _dx *
        (
            Vy[i1, j, ki] - Vy[i, j, ki] + Vy[i2, j, ki] - Vy[i1, j, ki] + Vy[i1, j1, ki] -
            Vy[i, j1, ki] + Vy[i2, j1, ki] - Vy[i1, j1, ki]
        )
    ∂Vy∂z =
        0.25 *
        _dz *
        (
            Vy[i1, j, k1] - Vy[i, j, k] + Vy[i2, j, k2] - Vy[i1, j, k1] + Vy[i1, j1, k1] -
            Vy[i, j1, k] + Vy[i2, j1, k2] - Vy[i1, j1, k1]
        )
    ∂Vz∂x =
        0.25 *
        _dx *
        (
            Vz[i1, j, k] - Vz[i, j, k] + Vz[i2, j, k] - Vz[i1, j, k] + Vz[i1, j1, k1] -
            Vz[i, j1, 1k] + Vz[i2, j1, k1] - Vz[i1, j1, 1k]
        )
    ∂Vz∂y =
        0.25 *
        _dy *
        (
            Vz[i1, j, k] - Vz[i, j, k] + Vz[i2, j, k] - Vz[i1, j, k] + Vz[i1, j1, k1] -
            Vz[i, j1, k1] + Vz[i2, j1, k1] - Vz[i1, j1, k1]
        )
    return ∂Vx∂y, ∂Vx∂z, ∂Vy∂x, ∂Vy∂z, ∂Vz∂x, ∂Vz∂y
end

Base.@propagate_inbounds @inline function compute_vorticity(∂V∂x::NTuple{2,T}) where {T}
    return ∂V∂x[2] - ∂V∂x[1]
end # 2D
Base.@propagate_inbounds @inline function compute_vorticity(∂V∂x::NTuple{3,T}) where {T}
    return ∂V∂x[3] - ∂V∂x[2], ∂V∂x[1] - ∂V∂x[3], ∂V∂x[2] - ∂V∂x[1]
end # 3D
