@parallel_indices (i, j) function rotate_stress!(V, τ::NTuple{3,T}, _di, dt) where {T}
    @inbounds rotate_stress!(V, τ, (i, j), _di, dt)
    return nothing
end

@parallel_indices (i, j, k) function rotate_stress!(V, τ::NTuple{6,T}, _di, dt) where {T}
    @inbounds rotate_stress!(V, τ, (i, j, k), _di, dt)
    return nothing
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
    τ_voigt = ntuple(Val(N)) do k
        Base.@_inline_meta
        τ[k][idx...]
    end
    # actually rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ω, τ_voigt, dt)

    ## 3) Update stress
    for k in 1:N
        τ[k][idx...] = muladd(τij_adv[k], dt, τr_voigt[k])
    end
    return nothing
end

# 2D
Base.@propagate_inbounds function advect_stress(τxx, τyy, τxy, Vx, Vy, i, j, _dx, _dy)
    τ = τxx, τyy, τxy
    τ_adv = ntuple(Val(3)) do k
        Base.@_inline_meta
        dx_right, dx_left, dy_up, dy_down = upwind_derivatives(τ[k], i, j)
        advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
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
        advection_term(
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
