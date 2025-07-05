using MuladdMacro
# check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
# @inline limit_periodic(a, n) = a > n ? n : (a < 1 ? 1 : a)
@inline function limit_periodic(a, n)
    a > n && return n
    a < 1 && return 1
    return a
end

## Betas
@inline function weno_betas(u1, u2, u3, u4, u5, weno)
    β0 = @muladd weno.c1 * (u1 - 2 * u2 + u3)^2 + weno.c2 * (u1 - 4 * u2 + 3 * u3)^2
    β1 = @muladd weno.c1 * (u2 - 2 * u3 + u4)^2 + weno.c2 * (u2 - u4)^2
    β2 = @muladd weno.c1 * (u3 - 2 * u4 + u5)^2 + weno.c2 * (3 * u3 - 4 * u4 + u5)^2
    return β0, β1, β2
end

## Upwind alphas
@inline function weno_alphas_upwind(::JustRelax.WENO5, ::Type{Any}, β0, β1, β2)
    return error("Unknown method for the WENO Scheme")
end

@inline function weno_alphas_upwind(weno::JustRelax.WENO5, ::Val{1}, β0, β1, β2)
    α0L = weno.d0L * inv(β0 + weno.ϵ)^2
    α1L = weno.d1L * inv(β1 + weno.ϵ)^2
    α2L = weno.d2L * inv(β2 + weno.ϵ)^2
    return α0L, α1L, α2L
end

@inline function weno_alphas_upwind(weno::JustRelax.WENO5, ::Val{2}, β0, β1, β2)
    τ = abs(β0 - β2)
    α0L = weno.d0L * (1 + (τ * inv(β0 + weno.ϵ))^2)
    α1L = weno.d1L * (1 + (τ * inv(β1 + weno.ϵ))^2)
    α2L = weno.d2L * (1 + (τ * inv(β2 + weno.ϵ))^2)
    return α0L, α1L, α2L
end

## Downwind alphas
@inline function weno_alphas_downwind(::JustRelax.WENO5, ::Any, β0, β1, β2)
    return error("Unknown method for the WENO Scheme")
end

@inline function weno_alphas_downwind(weno::JustRelax.WENO5, ::Val{1}, β0, β1, β2)
    α0R = weno.d0R * inv(β0 + weno.ϵ)^2
    α1R = weno.d1R * inv(β1 + weno.ϵ)^2
    α2R = weno.d2R * inv(β2 + weno.ϵ)^2
    return α0R, α1R, α2R
end

@inline function weno_alphas_downwind(weno::JustRelax.WENO5, ::Val{2}, β0, β1, β2)
    τ = abs(β0 - β2)
    α0R = weno.d0R * (1 + (τ * inv(β0 + weno.ϵ))^2)
    α1R = weno.d1R * (1 + (τ * inv(β1 + weno.ϵ))^2)
    α2R = weno.d2R * (1 + (τ * inv(β2 + weno.ϵ))^2)
    return α0R, α1R, α2R
end

## Stencil candidates
@inline function stencil_candidate_upwind(u1, u2, u3, u4, u5, weno)
    s0 = @muladd weno.sc1 * u1 - weno.sc2 * u2 + weno.sc3 * u3
    s1 = @muladd -weno.sc4 * u2 + weno.sc5 * u3 + weno.sc1 * u4
    s2 = @muladd weno.sc1 * u3 + weno.sc5 * u4 - weno.sc4 * u5
    return s0, s1, s2
end

@inline function stencil_candidate_downwind(u1, u2, u3, u4, u5, weno)
    s0 = @muladd -weno.sc4 * u1 + weno.sc5 * u2 + weno.sc1 * u3
    s1 = @muladd weno.sc1 * u2 + weno.sc5 * u3 - weno.sc4 * u4
    s2 = @muladd weno.sc3 * u3 - weno.sc2 * u4 + weno.sc1 * u5
    return s0, s1, s2
end

# UP/DOWN-WIND FLUXES
@inline function WENO_u_downwind(u1, u2, u3, u4, u5, weno)
    return _WENO_u(
        u1, u2, u3, u4, u5, weno, weno_alphas_downwind, stencil_candidate_downwind
    )
end

@inline function WENO_u_upwind(u1, u2, u3, u4, u5, weno)
    return _WENO_u(u1, u2, u3, u4, u5, weno, weno_alphas_upwind, stencil_candidate_upwind)
end

@inline function _WENO_u(
        u1, u2, u3, u4, u5, weno, fun_alphas::F1, fun_stencil::F2
    ) where {F1, F2}

    # Smoothness indicators
    β = weno_betas(u1, u2, u3, u4, u5, weno)

    # classical approach
    α0, α1, α2 = fun_alphas(weno, weno.method, β...)

    _α = inv(α0 + α1 + α2)
    w0 = α0 * _α
    w1 = α1 * _α
    w2 = α2 * _α

    # Candidate stencils
    s0, s1, s2 = fun_stencil(u1, u2, u3, u4, u5, weno)

    # flux down/up-wind
    f = @muladd w0 * s0 + w1 * s1 + w2 * s2

    return f
end

# FLUXES

## x-axis
@inline function WENO_flux_downwind_x(u, nx, weno, i, j)
    return _WENO_flux_x(u, nx, weno, i, j, WENO_u_downwind)
end
@inline function WENO_flux_upwind_x(u, nx, weno, i, j)
    return _WENO_flux_x(u, nx, weno, i, j, WENO_u_upwind)
end

@inline function _WENO_flux_x(u, nx, weno, i, j, fun::F) where {F}
    jw, jww = clamp(j - 1, 1, nx), clamp(j - 2, 1, nx)
    je, jee = clamp(j + 1, 1, nx), clamp(j + 2, 1, nx)
    @inbounds begin
        u1 = u[i, jww]
        u2 = u[i, jw]
        u3 = u[i, j]
        u4 = u[i, je]
        u5 = u[i, jee]
    end
    return f = fun(u1, u2, u3, u4, u5, weno)
end

## y-axis
@inline function WENO_flux_downwind_y(u, ny, weno, i, j)
    return _WENO_flux_y(u, ny, weno, i, j, WENO_u_downwind)
end
@inline function WENO_flux_upwind_y(u, ny, weno, i, j)
    return _WENO_flux_y(u, ny, weno, i, j, WENO_u_upwind)
end

@inline function _WENO_flux_y(u, ny, weno, i, j, fun::F) where {F}
    iw, iww = clamp(i - 1, 1, ny), clamp(i - 2, 1, ny)
    ie, iee = clamp(i + 1, 1, ny), clamp(i + 2, 1, ny)

    @inbounds begin
        u1 = u[iww, j]
        u2 = u[iw, j]
        u3 = u[i, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
    end
    return f = fun(u1, u2, u3, u4, u5, weno)
end

@inline function weno_rhs(vy, vx, weno, _dx, _dy, nx, ny, i, j)
    jW, jE = clamp(j - 1, 1, ny), clamp(j + 1, 1, ny)
    iS, iN = clamp(i - 1, 1, nx), clamp(i + 1, 1, nx)

    return @inbounds begin
        vx_ij = vx[i, j]
        vy_ij = vy[i, j]

        r = @muladd max(vx_ij, 0) * (weno.fL[i, j] - weno.fL[i, jW]) * _dx +
            min(vx_ij, 0) * (weno.fR[i, jE] - weno.fR[i, j]) * _dx +
            max(vy_ij, 0) * (weno.fB[i, j] - weno.fB[iS, j]) * _dy +
            min(vy_ij, 0) * (weno.fT[iN, j] - weno.fT[i, j]) * _dy
    end
end

@parallel_indices inbounds = true (i, j) function weno_f!(u, weno, nx, ny)
    weno.fL[i, j] = WENO_flux_upwind_x(u, ny, weno, i, j)
    weno.fR[i, j] = WENO_flux_downwind_x(u, ny, weno, i, j)
    weno.fB[i, j] = WENO_flux_upwind_y(u, nx, weno, i, j)
    weno.fT[i, j] = WENO_flux_downwind_y(u, nx, weno, i, j)
    return nothing
end

## WENO-5 ADVECTION
"""
    WENO_advection!(u, Vxi, weno, di, ni, dt)

Perform the advection step of the Weighted Essentially Non-Oscillatory (WENO) scheme for the solution of hyperbolic partial differential equations.

# Arguments
- `u`: field to be advected.
- `Vxi`: velocity field.
- `weno`: structure containing the WENO scheme parameters and temporary variables.
- `di`: grid spacing.
- `ni`: number of grid points.
- `dt`: time step.

# Description
The function first calculates the fluxes using the WENO scheme. Then it performs three steps of the WENO scheme. Each step involves calculating the right-hand side of the WENO equation and updating the solution `u`. The updating of the solution `u` is done using different combinations of the original solution and the temporary solution `weno.ut`.
"""
function WENO_advection!(u, Vxi, weno, di, dt)
    _di = inv.(di)
    ni = nx, ny = size(u)
    one_third = inv(3)
    two_thirds = 2 * one_third

    @parallel (1:nx, 1:ny) weno_f!(u, weno, nx, ny)
    @parallel (1:nx, 1:ny) weno_step1!(weno, u, Vxi, _di, ni, dt)

    @parallel (1:nx, 1:ny) weno_f!(weno.ut, weno, nx, ny)
    @parallel (1:nx, 1:ny) weno_step2!(weno, u, Vxi, _di, ni, dt)

    @parallel (1:nx, 1:ny) weno_f!(weno.ut, weno, nx, ny)
    return @parallel (1:nx, 1:ny) weno_step3!(u, weno, Vxi, _di, ni, dt, one_third, two_thirds)
end

@parallel_indices (i, j) function weno_step1!(weno, u, Vxi, _di, ni, dt)
    rᵢ = weno_rhs(Vxi..., weno, _di..., ni..., i, j)
    @inbounds weno.ut[i, j] = muladd(-dt, rᵢ, u[i, j])
    return nothing
end

@parallel_indices (i, j) function weno_step2!(weno, u, Vxi, _di, ni, dt)
    rᵢ = weno_rhs(Vxi..., weno, _di..., ni..., i, j)
    @inbounds weno.ut[i, j] = @muladd 0.75 * u[i, j] + 0.25 * weno.ut[i, j] - 0.25 * dt * rᵢ
    return nothing
end

@parallel_indices (i, j) function weno_step3!(
        u, weno, Vxi, _di, ni, dt, one_third, two_thirds
    )
    rᵢ = weno_rhs(Vxi..., weno, _di..., ni..., i, j)
    @inbounds u[i, j] = @muladd one_third * u[i, j] + two_thirds * weno.ut[i, j] -
        two_thirds * dt * rᵢ
    return nothing
end

## WENO5-HYBRID advection as described in the paper:
#`Hybrid scheme for complex flows on staggered grids and application to multiphase flows`
# https://doi.org/10.1016/j.jcp.2018.12.041
"""
    muscl_interpolate(uL, uC, uR)

MUSCL TVD interpolation from cell centers to face.
"""
function muscl_interpolate(uL, uC, uR)
    ΔL = uC - uL
    ΔR = uR - uC
    slope = sign(ΔL + ΔR) * max(0.0, min(abs(ΔL), abs(ΔR)))
    return uC + 0.5 * slope
end

"""
    central_flux(uL, uR)

Simple second-order central flux.
"""
central_flux(uL, uR) = 0.5 * (uL + uR)

"""
    shock_sensor(uL, uC, uR, threshold)

Simple gradient-based shock detector.
"""
shock_sensor(uL, uC, uR, threshold) = abs(uR - uL) > threshold


"""
    staggered_hybrid_flux_x(u, nx, weno, i, j, threshold)

Compute hybrid staggered flux in x-direction.
"""
function staggered_hybrid_flux_x(u, nx, weno, i, j, threshold)
    jw, jww = clamp(j - 1, 1, nx), clamp(j - 2, 1, nx)
    je, jee = clamp(j + 1, 1, nx), clamp(j + 2, 1, nx)

    @inbounds begin
        u1 = u[i, jww]
        u2 = u[i, jw]
        u3 = u[i, j]
        u4 = u[i, je]
        u5 = u[i, jee]
    end

    # WENO flux
    flux_weno = WENO_u_upwind(u1, u2, u3, u4, u5, weno)

    # Central flux: simple average between j and j+1
    flux_central = central_flux(u[i, j], u[i, je])

    # Shock detection
    is_shock = shock_sensor(u2, u3, u4, threshold)

    # Hybrid weight
    w = is_shock ? 1.0 : 0.0

    # Blend
    return (1 - w)*flux_central + w*flux_weno
end

# -----------------------------
# Staggered hybrid flux in y-direction
# -----------------------------
"""
    staggered_hybrid_flux_y(u, ny, weno, i, j, threshold)

Compute hybrid staggered flux in y-direction.
"""
function staggered_hybrid_flux_y(u, ny, weno, i, j, threshold)
    iw, iww = clamp(i - 1, 1, ny), clamp(i - 2, 1, ny)
    ie, iee = clamp(i + 1, 1, ny), clamp(i + 2, 1, ny)

    @inbounds begin
        u1 = u[iww, j]
        u2 = u[iw, j]
        u3 = u[i, j]
        u4 = u[ie, j]
        u5 = u[iee, j]
    end

    # WENO flux
    flux_weno = WENO_u_upwind(u1, u2, u3, u4, u5, weno)

    # Central flux: average between i and i+1
    flux_central = central_flux(u[i, j], u[ie, j])

    # Shock detection
    is_shock = shock_sensor(u2, u3, u4, threshold)

    # Hybrid weight
    w = is_shock ? 1.0 : 0.0

    # Blend
    return (1 - w)*flux_central + w*flux_weno
end

# -----------------------------
# Compute hybrid staggered fluxes in all directions
# -----------------------------
"""
    staggered_weno_f!(u, weno, nx, ny, threshold)

Compute hybrid staggered fluxes in all directions.
"""
@parallel_indices inbounds = true (i, j) function staggered_weno_f!(u, weno, nx, ny, threshold)
    weno.fL[i, j] = staggered_hybrid_flux_x(u, ny, weno, i, j, threshold)
    weno.fR[i, j] = staggered_hybrid_flux_x(u, ny, weno, i, j+1, threshold)
    weno.fB[i, j] = staggered_hybrid_flux_y(u, nx, weno, i, j, threshold)
    weno.fT[i, j] = staggered_hybrid_flux_y(u, nx, weno, i+1, j, threshold)
    return nothing
end

# -----------------------------
# New main advection function
# -----------------------------
"""
    WENO_advection_staggered!(u, Vxi, weno, di, dt)

Perform staggered grid hybrid WENO-central advection.
"""
function WENO_advection_staggered!(u, Vxi, weno, di, dt)
    _di = inv.(di)
    ni = nx, ny = size(u)
    one_third = inv(3)
    two_thirds = 2 * one_third
    threshold = 1e-3  # You can adjust this threshold

    @parallel (1:nx, 1:ny) staggered_weno_f!(u, weno, nx, ny, threshold)
    @parallel (1:nx, 1:ny) weno_step1!(weno, u, Vxi, _di, ni, dt)

    @parallel (1:nx, 1:ny) staggered_weno_f!(weno.ut, weno, nx, ny, threshold)
    @parallel (1:nx, 1:ny) weno_step2!(weno, u, Vxi, _di, ni, dt)

    @parallel (1:nx, 1:ny) staggered_weno_f!(weno.ut, weno, nx, ny, threshold)
    return @parallel (1:nx, 1:ny) weno_step3!(u, weno, Vxi, _di, ni, dt, one_third, two_thirds)
end

const JR_T = Union{
    StokesArrays,
    SymmetricTensor,
    ThermalArrays,
    Velocity,
    Displacement,
    Vorticity,
    Residual,
    Viscosity,
}

"""
    WENO_advection_staggered!(u::T, Vxi, weno, di, dt)

Apply staggered WENO advection to all fields of `u`.
"""
@generated function WENO_advection_staggered!(
    u::T, Vxi, weno, di, dt
) where {T <: JR_T}
    nfields = fieldcount(T)
    exprs = Expr[]
    for i in 1:nfields
        push!(exprs, quote
            field = getfield(u, $i)
            if field !== nothing
                WENO_advection_staggered!(field, Vxi, weno, di, dt)
            end
        end)
    end
    return quote
        $(exprs...)
        return nothing
    end
end
