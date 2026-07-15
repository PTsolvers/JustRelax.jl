## VARIATIONAL (masked) DYREL VELOCITY KERNELS
#
# Masked (RockRatio ϕ) counterparts of the fused DYREL kernels in velocity_kernels.jl, used by the
# variational `_solve_DYREL!` (solver_VS.jl). 2D only — the variational DYREL path is 2D-only.

## DIVERGENCE + DEVIATORIC STRAIN RATE + PRESSURE RESIDUAL (fused, masked)
# Masked analogue of `compute_∇V_strain_rate_RP!`: divergence, deviatoric strain rate and the
# pressure residual RP are computed in a single pass over the (rock) valid cells, reusing the
# in-register divergence instead of reading ∇V back. As in the non-variational fused kernel, ∇V
# itself is NOT stored inside the loop (RP is derived from the in-register `div_ij`); the public
# `stokes.∇V` diagnostic is recomputed once from the converged velocity field after the loop.
# ε.*_c interpolation is likewise skipped in-loop (the stress kernel reads ε.xy at vertices).
function compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ::JustRelax.RockRatio, _di, ni, dt, args)
    ΔT = haskey(args, :ΔT) ? args.ΔT : nothing
    melt_fraction = haskey(args, :melt_fraction) ? args.melt_fraction : nothing
    @parallel (@idx ni .+ 1) compute_∇V_strain_rate_RP!(
        @strain(stokes)...,
        @velocity(stokes)...,
        stokes.R.RP,
        stokes.P,
        stokes.P0,
        stokes.Q,
        dyrel.ηb,
        ϕ,
        _di.vertex,
        _di.velocity...,
        rheology,
        phase_ratios.center,
        ΔT,
        melt_fraction,
        dt,
    )
    return nothing
end

@parallel_indices (i, j) function compute_∇V_strain_rate_RP!(
        εxx::AbstractArray{T, 2},
        εyy,
        εxy,
        Vx,
        Vy,
        RP,
        P,
        P0,
        Q,
        ηb,
        ϕ::JustRelax.RockRatio,
        _di_vertex,
        _di_vx,
        _di_vy,
        rheology,
        phase_ratio,
        ΔT,
        melt_fraction,
        dt,
    ) where {T}

    third = T(1) / T(3)

    @inbounds begin
        vx_s = Vx[i, j]
        vx_n = Vx[i, j + 1]
        vy_w = Vy[i, j]
        vy_e = Vy[i + 1, j]

        if i ≤ size(εxy, 1) && j ≤ size(εxy, 2)
            if isvalid_v(ϕ, i, j)
                _dy_vx = @dy(_di_vx, j)
                _dx_vy = @dx(_di_vy, i)
                dVx_dy = (vx_n - vx_s) * _dy_vx
                dVy_dx = (vy_e - vy_w) * _dx_vy
                εxy[i, j] = 0.5 * (dVx_dy + dVy_dx)
            else
                εxy[i, j] = zero(T)
            end
        end

        if i ≤ size(εxx, 1) && j ≤ size(εxx, 2)
            if isvalid_c(ϕ, i, j)
                vx_ne = Vx[i + 1, j + 1]
                vy_ne = Vy[i + 1, j + 1]
                _dx, _dy = @dxi(_di_vertex, i, j)

                dVx_dx = (vx_ne - vx_n) * _dx
                dVy_dy = (vy_ne - vy_e) * _dy
                div_ij = dVx_dx + dVy_dy

                div_third = div_ij * third
                εxx[i, j] = dVx_dx - div_third
                εyy[i, j] = dVy_dy - div_third

                # fused pressure residual (reuses `div_ij` in-register). Masked to 0 in air below,
                # where ηb = 0 would make (P - P0)/ηb NaN.
                RP[i, j] = _RP_cell(P[i, j], P0[i, j], div_ij, Q[i, j], ηb[i, j], dt, rheology, phase_ratio, ΔT, melt_fraction, i, j)
            else
                εxx[i, j] = zero(T)
                εyy[i, j] = zero(T)
                RP[i, j] = zero(T)
            end
        end
    end

    return nothing
end

## RESIDUALS (masked)

@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        P,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vertex,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            _dx_c = @dx(_di_center, i)
            _dy_v = @dy(_di_vertex, j)
            Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
            Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
            Rx[i, j] = if isvalid_vx(ϕ, i + 1, j)
                d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - d_xa(P, ϕ.center) - d_xa(ΔPψ, ϕ.center) - av_xa(ρgx, ϕ.center)
            else
                0.0e0
            end
        end
        if all((i, j) .≤ size(Ry))
            _dy_c = @dy(_di_center, j)
            _dx_v = @dx(_di_vertex, i)
            Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
            Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
            Ry[i, j] = if isvalid_vy(ϕ, i, j + 1)
                d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - d_ya(P, ϕ.center) - d_ya(ΔPψ, ϕ.center) - av_ya(ρgy, ϕ.center)
            else
                0.0e0
            end
        end
    end
    return nothing
end

# Free-surface-stabilized variant of the masked PH residual: adds the ϕ-weighted implicit
# free-surface advection term (Vy·∂ρg∂y·dt) to the vertical residual, mirroring the variational
# `compute_Vy!` correction. `dt` is passed as `dt * free_surface` by the solver.
@parallel_indices (i, j) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        Vx,
        Vy,
        P,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vertex,
        dt,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)

    ny = size(ρgy, 2)
    @inbounds begin
        if all((i, j) .≤ size(Rx))
            _dx_c = @dx(_di_center, i)
            _dy_v = @dy(_di_vertex, j)
            Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
            Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
            Rx[i, j] = if isvalid_vx(ϕ, i + 1, j)
                d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - d_xa(P, ϕ.center) - d_xa(ΔPψ, ϕ.center) - av_xa(ρgx, ϕ.center)
            else
                0.0e0
            end
        end
        if all((i, j) .≤ size(Ry))
            _dy_c = @dy(_di_center, j)
            _dx_v = @dx(_di_vertex, i)
            Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
            Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
            Ry[i, j] = if isvalid_vy(ϕ, i, j + 1)
                # free-surface stabilization term (ρg masked by ϕ.center, as in `compute_Vy!`)
                θ = 1.0
                Vyᵢⱼ = Vy[i + 1, j + 1]
                j_N = min(j + 1, ny)
                ρg_S = ρgy[i, j] * ϕ.center[i, j]
                ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
                ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
                ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt

                d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - d_ya(P, ϕ.center) - d_ya(ΔPψ, ϕ.center) - av_ya(ρgy, ϕ.center) + ρg_correction
            else
                0.0e0
            end
        end
    end
    return nothing
end

## DR VELOCITY RESIDUAL + DAMPED UPDATE (fused, masked)
@parallel_indices (i, j) function compute_DR_residual_update_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        Vx,
        Vy,
        dVxdτ,
        dVydτ,
        P,
        θc,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        Dx,
        Dy,
        αVx,
        αVy,
        βVx,
        βVy,
        dτVx,
        dτVy,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vertex,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            _dx_c = @dx(_di_center, i)
            _dy_v = @dy(_di_vertex, j)
            Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
            Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
            if isvalid_vx(ϕ, i + 1, j)
                Rx_ij = (d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - d_xa(P, ϕ.center) - d_xa(θc, ϕ.center) - av_xa(ρgx, ϕ.center)) / Dx[i, j]
                Rx[i, j] = Rx_ij
                dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij, αVx[i, j], βVx[i, j], dτVx[i, j])
                dVxdτ[i, j] = dVx_new
                Vx[i + 1, j + 1] += ΔVx
            else
                Rx[i, j] = zero(T)
                dVxdτ[i, j] = zero(T)
                Vx[i + 1, j + 1] = zero(T)
            end
        end
        if all((i, j) .≤ size(Ry))
            _dy_c = @dy(_di_center, j)
            _dx_v = @dx(_di_vertex, i)
            Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
            Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
            if isvalid_vy(ϕ, i, j + 1)
                Ry_ij = (d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - d_ya(P, ϕ.center) - d_ya(θc, ϕ.center) - av_ya(ρgy, ϕ.center)) / Dy[i, j]
                Ry[i, j] = Ry_ij
                dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij, αVy[i, j], βVy[i, j], dτVy[i, j])
                dVydτ[i, j] = dVy_new
                Vy[i + 1, j + 1] += ΔVy
            else
                Ry[i, j] = zero(T)
                dVydτ[i, j] = zero(T)
                Vy[i + 1, j + 1] = zero(T)
            end
        end
    end

    return nothing
end

# Free-surface-stabilized variant of the masked fused DR kernel: adds the ϕ-weighted implicit
# free-surface advection term (Vy·∂ρg∂y·dt) to the vertical momentum residual, mirroring the FS
# `compute_PH_residual_V!` overload above. The solver passes `dt * free_surface`.
@parallel_indices (i, j) function compute_DR_residual_update_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        Vx,
        Vy,
        dVxdτ,
        dVydτ,
        P,
        θc,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        Dx,
        Dy,
        αVx,
        αVy,
        βVx,
        βVy,
        dτVx,
        dτVy,
        ϕ::JustRelax.RockRatio,
        _di_center,
        _di_vertex,
        dt,
    ) where {T}
    Base.@propagate_inbounds @inline av_xa(A, ϕ) = _av_xa(A, ϕ, i, j)
    Base.@propagate_inbounds @inline av_ya(A, ϕ) = _av_ya(A, ϕ, i, j)

    ny = size(ρgy, 2)
    @inbounds begin
        if all((i, j) .≤ size(Rx))
            _dx_c = @dx(_di_center, i)
            _dy_v = @dy(_di_vertex, j)
            Base.@propagate_inbounds @inline d_xa(A, ϕ) = _d_xa(A, ϕ, _dx_c, i, j)
            Base.@propagate_inbounds @inline d_yi(A, ϕ) = _d_yi(A, ϕ, _dy_v, i, j)
            if isvalid_vx(ϕ, i + 1, j)
                Rx_ij = (d_xa(τxx, ϕ.center) + d_yi(τxy, ϕ.vertex) - d_xa(P, ϕ.center) - d_xa(θc, ϕ.center) - av_xa(ρgx, ϕ.center)) / Dx[i, j]
                Rx[i, j] = Rx_ij
                dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij, αVx[i, j], βVx[i, j], dτVx[i, j])
                dVxdτ[i, j] = dVx_new
                Vx[i + 1, j + 1] += ΔVx
            else
                Rx[i, j] = zero(T)
                dVxdτ[i, j] = zero(T)
                Vx[i + 1, j + 1] = zero(T)
            end
        end
        if all((i, j) .≤ size(Ry))
            _dy_c = @dy(_di_center, j)
            _dx_v = @dx(_di_vertex, i)
            Base.@propagate_inbounds @inline d_ya(A, ϕ) = _d_ya(A, ϕ, _dy_c, i, j)
            Base.@propagate_inbounds @inline d_xi(A, ϕ) = _d_xi(A, ϕ, _dx_v, i, j)
            if isvalid_vy(ϕ, i, j + 1)
                # free-surface stabilization term (ρg masked by ϕ.center, as in `compute_Vy!`)
                θ = 1.0
                Vyᵢⱼ = Vy[i + 1, j + 1]
                j_N = min(j + 1, ny)
                ρg_S = ρgy[i, j] * ϕ.center[i, j]
                ρg_N = ρgy[i, j_N] * ϕ.center[i, j_N]
                ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
                ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt

                Ry_ij = (d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - d_ya(P, ϕ.center) - d_ya(θc, ϕ.center) - av_ya(ρgy, ϕ.center) + ρg_correction) / Dy[i, j]
                Ry[i, j] = Ry_ij
                dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij, αVy[i, j], βVy[i, j], dτVy[i, j])
                dVydτ[i, j] = dVy_new
                Vy[i + 1, j + 1] += ΔVy
            else
                Ry[i, j] = zero(T)
                dVydτ[i, j] = zero(T)
                Vy[i + 1, j + 1] = zero(T)
            end
        end
    end

    return nothing
end
