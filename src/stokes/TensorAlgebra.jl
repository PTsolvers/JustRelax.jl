
using GeoParams, ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

"""
    Jaumann derivative

τij_o += v_k * ∂τij_o/∂x_k - ω_ij * ∂τkj_o + ∂τkj_o * ω_ij

"""
# original 2D version
@parallel_indices (i, j) function rotate_stress(Vx, Vy, τxx, τyy, τxy, _dx, _dy, dt)

    ## 1) Advect stress
    @inbounds Vxᵢⱼ = 0.5 * (Vx[i    , j + 1] + Vx[i + 1, j + 1]) # averages @ cell center
    @inbounds Vyᵢⱼ = 0.5 * (Vy[i + 1, j    ] + Vy[i + 1, j + 1]) # averages @ cell center
    τxx_adv, τyy_adv, τxy_adv = @inline advect_stress(τxx, τyy, τxy, Vxᵢⱼ, Vyᵢⱼ, i, j, _dx, _dy)

    ## 2) Rotate stress
    # average ∂Vx/∂y @ cell center
    @inbounds ∂Vx∂y = 0.25 * _dy * (
        Vx[1, 2] - Vx[1, 1] + 
        Vx[1, 3] - Vx[1, 2] + 
        Vx[2, 2] - Vx[2, 1] + 
        Vx[2, 3] - Vx[2, 2]
    )
    # average ∂Vy/∂x @ cell center
   @inbounds  ∂Vy∂x = 0.25 * _dx * (
        Vy[2, 1] - Vy[1, 1] + 
        Vy[3, 1] - Vy[2, 1] + 
        Vy[2, 2] - Vy[1, 2] + 
        Vy[3, 2] - Vy[2, 2]
    )
    # compute xy component of the vorticity tensor; normal components = 0.0
    ωxy = 0.5 * (∂Vx∂y - ∂Vy∂x)
    # stress tensor in Voigt notation
    @inbounds τ_voigt = τxx[i, j], τyy[i, j], τxy[i, j]
    # rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ωxy, τ_voigt, dt)
    
    ## 3) Update stress
    @inbounds τxx[i, j] += τr_voigt[1] + τxx_adv
    @inbounds τyy[i, j] += τr_voigt[2] + τyy_adv
    @inbounds τxy[i, j] += τr_voigt[3] + τxy_adv
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
    return (Vx > 0) * Vx * dx_right * _dx +
        (Vx < 0) * Vx * dx_left * _dx +
        (Vy > 0) * Vy * dy_up * _dy +
        (Vy < 0) * Vy * dy_down * _dy
end

function upwind_derivatives(A, i, j)
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


#################

@parallel_indices (i, j) function rotate_stress!(V, τ::NTuple{3, T}, _di, dt) where T
    @inline rotate_stress!(V, τ, (i, j) , _di, dt)
    return nothing
end

@parallel_indices (i, j, k) function rotate_stress!(V, τ::NTuple{6, T}, _di, dt) where T
    @inline rotate_stress!(V, τ, (i, j, k) , _di, dt)
    return nothing
end

function rotate_stress!(V, τ::NTuple{N, T}, idx, _di, dt) where {N, T}

    ## 1) Advect stress
    Vᵢⱼ = @inline velocity2center(V..., idx...) # averages @ cell center
    τij_adv = @inline advect_stress(τ..., Vᵢⱼ..., idx..., _di...)

    ## 2) Rotate stress
    # average ∂Vx/∂y @ cell center
    ∂V∂x = ∂Vi∂xi(V..., _di..., idx...)
    # compute xy component of the vorticity tensor; normal components = 0.0
    ω = compute_vorticity(∂V∂x)
    # stress tensor in Voigt notation
    τ_voigt = ntuple(Val(N)) do k
        Base.@_inline_meta
        @inbounds τ[k][idx...]
    end
    # rotate stress tensor
    τr_voigt = GeoParams.rotate_elastic_stress2D(ω, τ_voigt, dt)
    
    ## 3) Update stress
    # worth unrolling?
    for k in 1:N
        @inbounds τ[k][idx...] += τr_voigt[k] + τij_adv[k]
    end

    return 
end

@inline function velocity2center(Vx, Vy, i, j)
    @inbounds Vxᵢⱼ = 0.5 * (Vx[i    , j + 1] + Vx[i + 1, j + 1]) # averages @ cell center
    @inbounds Vyᵢⱼ = 0.5 * (Vy[i + 1, j    ] + Vy[i + 1, j + 1]) # averages @ cell center
    return Vxᵢⱼ, Vyᵢⱼ
end

@inline function ∂Vi∂xi(Vx, Vy, _dx, _dy, i, j)
    @inbounds ∂Vx∂y = 0.25 * _dy * (
        Vx[i  , j+1] - Vx[i  , 1  ] + 
        Vx[i  , j+2] - Vx[i  , j+1] + 
        Vx[i+1, j+1] - Vx[i+1, 1  ] + 
        Vx[i+1, j+2] - Vx[i+1, j+1]
    )
    # average ∂Vy/∂x @ cell center
   @inbounds  ∂Vy∂x = 0.25 * _dx * (
        Vy[i+1, j  ] - Vy[i  , j  ] + 
        Vy[i+2, j  ] - Vy[i+1, j  ] + 
        Vy[i+1, j+1] - Vy[i  , j+1] + 
        Vy[i+2, j+1] - Vy[i+1, j+1]
    )
    return ∂Vx∂y, ∂Vy∂x
end

@inline compute_vorticity(∂V∂x::NTuple{2, T}) where T = 0.5 * (∂V∂x[1] - ∂V∂x[2])

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


@btime @parallel $(1:n, 1:n) rotate_stress!($V, $τ, $_di, $dt)
@btime @parallel $(1:n, 1:n) rotate_stress($Vx, $Vy, $txx, $tyy, $txy, $(_di[1]), $(_di[2]), $dt)

ProfileCanvas.@profview for i in 1:100 @parallel (1:n, 1:n) rotate_stress(Vx, Vy, txx, tyy, txy, _di[1], _di[2], dt) end
ProfileCanvas.@profview for i in 1:100 @parallel (1:n, 1:n) rotate_stress!(V, τ, _di, dt) end