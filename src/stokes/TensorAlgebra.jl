@parallel_indices (i, j) function rotate_stress!(V, τ::NTuple{3, T}, _di, dt) where T
    @inline rotate_stress!(V, τ, (i, j) , _di, dt)
    return nothing
end

@parallel_indices (i, j, k) function rotate_stress!(V, τ::NTuple{6, T}, _di, dt) where T
    @inline rotate_stress!(V, τ, (i, j, k) , _di, dt)
    return nothing
end

"""
    Jaumann derivative

τij_o += v_k * ∂τij_o/∂x_k - ω_ij * ∂τkj_o + ∂τkj_o * ω_ij

"""
function rotate_stress!(V, τ::NTuple{N, T}, idx, _di, dt) where {N, T}

    ## 1) Advect stress
    Vᵢⱼ = @inbounds velocity2center(V..., idx...) # averages @ cell center
    τij_adv = @inbounds advect_stress(τ..., Vᵢⱼ..., idx..., _di...)

    ## 2) Rotate stress
    # average ∂Vx/∂y @ cell center
    ∂V∂x = @inbounds cross_derivatives(V..., _di..., idx...)
    # compute xy component of the vorticity tensor; normal components = 0.0
    ω = @inbounds compute_vorticity(∂V∂x)
    # stress tensor in Voigt notation
    τ_voigt = ntuple(Val(N)) do k
        Base.@_inline_meta
        @inbounds τ[k][idx...]
    end
    # rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ω, τ_voigt, dt)
    
    ## 3) Update stress
    for k in 1:N    
        @inbounds τ[k][idx...] += muladd(τij_adv[k], dt, τr_voigt[k])
        # @inbounds τ[k][idx...] += dt * (τr_voigt[k] + τij_adv[k])
    end
    return 
end

Base.@propagate_inbounds function advect_stress(τxx, τyy, τxy, Vx, Vy, i, j, _dx, _dy)
    τ = τxx, τyy, τxy
    τ_adv = ntuple(Val(3)) do k
        Base.@_inline_meta
        dx_right, dx_left, dy_up, dy_down = upwind_derivatives(τ[k], i, j)
        advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    end
    return τ_adv
end

function advection_term(Vx, Vy, dx_right, dx_left, dy_up, dy_down, _dx, _dy)
    return (Vx > 0 ? dx_right : dx_left) * Vx * _dx +
           (Vy > 0 ? dy_up : dy_down) * Vy * _dy
end

Base.@propagate_inbounds function upwind_derivatives(A, i, j)
    nx, ny =  size(A)

    # dx derivatives
    x_left   = i - 1 > 1 ? A[i-1, j] : 0.0
    x_center = A[i, j]
    x_right  = i + 1 < nx ? A[i+1, j] : 0.0
    dx_right = x_right  - x_center
    dx_left  = x_center - x_left
    
    # dy derivatives
    y_down   = j - 1 > 1 ? A[i, j-1] : 0.0
    y_center = A[i, j]
    y_up     = j + 1 < ny ? A[i, j+1] : 0.0
    dy_up    = y_up - y_center
    dy_down  = y_center - y_down

    return dx_right, dx_left, dy_up, dy_down
end

# averages @ cell center 2D
Base.@propagate_inbounds @inline function velocity2center(Vx, Vy, i, j)
    i1, j1 = @add 1 i j
    Vxᵢⱼ = 0.5 * (Vx[i , j1] + Vx[i1, j1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j ] + Vy[i1, j1])
    return Vxᵢⱼ, Vyᵢⱼ
end

# averages @ cell center 3D
Base.@propagate_inbounds @inline function velocity2center(Vx, Vy, Vz, i, j, k)
    i1, j1, k1 = @add 1 i j k
    Vxᵢⱼ = 0.5 * (Vx[i , j1, k1] + Vx[i1, j1, k1])
    Vyᵢⱼ = 0.5 * (Vy[i1, j , k1] + Vy[i1, j1, k1])
    Vzᵢⱼ = 0.5 * (Vz[i1, j1, k ] + Vz[i1, j1, k1])
    return Vxᵢⱼ, Vyᵢⱼ, Vzᵢⱼ
end

# 2D
Base.@propagate_inbounds @inline function cross_derivatives(Vx, Vy, _dx, _dy, i, j)
    i1, j1 = @add 1 i j
    i2, j2 = @add 2 i j
    # average @ cell center
    ∂Vx∂y = 0.25 * _dy * (
        Vx[i , j1] - Vx[i , j ] + 
        Vx[i , j2] - Vx[i , j1] + 
        Vx[i1, j1] - Vx[i1, j ] + 
        Vx[i1, j2] - Vx[i1, j1]
    )
    ∂Vy∂x = 0.25 * _dx * (
        Vy[i1, j ] - Vy[i , j ] + 
        Vy[i2, j ] - Vy[i1, j ] + 
        Vy[i1, j1] - Vy[i , j1] + 
        Vy[i2, j1] - Vy[i1, j1]
    )
    return ∂Vx∂y, ∂Vy∂x
end

# TOFINISH 3D
Base.@propagate_inbounds @inline function cross_derivatives(Vx, Vy, Vz, _dx, _dy, _dz, i, j, k)
    i1, j1, k2 = @add 1 i j k
    i2, j2, k2 = @add 2 i j k
    # cross derivatives @ cell centers
    ∂Vx∂y = 0.25 * _dy * (
        Vx[i , j1, k] - Vx[i , j , k] + 
        Vx[i , j2, k] - Vx[i , j1, k] + 
        Vx[i1, j1, k] - Vx[i1, j , k] + 
        Vx[i1, j2, k] - Vx[i1, j1, k]
    )
    ∂Vx∂z = 0.25 * _dz * (
        Vx[i , j1, k] - Vx[i , j , k] + 
        Vx[i , j2, k] - Vx[i , j1, k] + 
        Vx[i1, j1, k] - Vx[i1, j , k] + 
        Vx[i1, j2, k] - Vx[i1, j1, k]
    )
    ∂Vy∂x = 0.25 * _dx * (
        Vy[i1, j , k] - Vy[i , j , k] + 
        Vy[i2, j , k] - Vy[i1, j , k] + 
        Vy[i1, j1, k] - Vy[i , j1, k] + 
        Vy[i2, j1, k] - Vy[i1, j1, k]
    )
    ∂Vy∂z = 0.25 * _dz * (
        Vy[i1, j , k] - Vy[i , j , k] + 
        Vy[i2, j , k] - Vy[i1, j , k] + 
        Vy[i1, j1, k] - Vy[i , j1, k] + 
        Vy[i2, j1, k] - Vy[i1, j1, k]
    )
    ∂Vz∂x = 0.25 * _dx * (
        Vz[i1, j , k] - Vz[i , j , k] + 
        Vz[i2, j , k] - Vz[i1, j , k] + 
        Vz[i1, j1, k] - Vz[i , j1, k] + 
        Vz[i2, j1, k] - Vz[i1, j1, k]
    )
    ∂Vz∂y = 0.25 * _dx * (
        Vz[i1, j , k] - Vz[i , j , k] + 
        Vz[i2, j , k] - Vz[i1, j , k] + 
        Vz[i1, j1, k] - Vz[i , j1, k] + 
        Vz[i2, j1, k] - Vz[i1, j1, k]
    )        
    return ∂Vx∂y, ∂Vx∂z, ∂Vy∂x, ∂Vy∂z, ∂Vz∂x, ∂Vz∂y
end

Base.@propagate_inbounds @inline compute_vorticity(∂V∂x::NTuple{2, T}) where T = ∂V∂x[1] - ∂V∂x[2] # 2D
Base.@propagate_inbounds @inline compute_vorticity(∂V∂x::NTuple{3, T}) where T = ∂V∂x[3] - ∂V∂x[2], ∂V∂x[1] - ∂V∂x[3], ∂V∂x[2] - ∂V∂x[1] # 3D

n = 128
Vx = rand(n+1,n+2)
Vy = rand(n+2,n+1)
txx= rand(n,n)
tyy= rand(n,n)
txy= rand(n,n)
V = Vx, Vy
τ = txx, tyy, txy
_di = rand(),rand()
dt = 1.0


# @btime @parallel $(1:n, 1:n) rotate_stress!($V, $τ, $_di, $dt)
# @btime @parallel $(1:n, 1:n) rotate_stress($Vx, $Vy, $txx, $tyy, $txy, $(_di[1]), $(_di[2]), $dt)

# ProfileCanvas.@profview for i in 1:100 @parallel (1:n, 1:n) rotate_stress(Vx, Vy, txx, tyy, txy, _di[1], _di[2], dt) end
# ProfileCanvas.@profview for i in 1:100 @parallel (1:n, 1:n) rotate_stress!(V, τ, _di, dt) end


# averages @ cell center 2D
# Base.@propagate_inbounds @inline function velocity2center(Vx, Vy, i, j)
#     Vxᵢⱼ = 0.5 * (Vx[i    , j + 1] + Vx[i + 1, j + 1])
#     Vyᵢⱼ = 0.5 * (Vy[i + 1, j    ] + Vy[i + 1, j + 1])
#     return Vxᵢⱼ, Vyᵢⱼ
# end

# Base.@propagate_inbounds @inline function velocity2center2(Vx, Vy, i, j)
#     i1, j1 = @add 1 i j
#     Vxᵢⱼ = 0.5 * (Vx[i , j1] + Vx[i1, j1])
#     Vyᵢⱼ = 0.5 * (Vy[i1, j ] + Vy[i1, j1])
#     return Vxᵢⱼ, Vyᵢⱼ
# end

# @benchmark velocity2center($Vx, $Vy, 2, 2)
# @benchmark velocity2center2($Vx, $Vy, 2, 2)