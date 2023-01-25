
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