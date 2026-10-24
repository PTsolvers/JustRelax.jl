# Three-dimensional reduced-space DYREL kernels. Stress and pressure weights
# enter the momentum operator once; eliminated rows never participate in DR.
@inline valid_stress(::Nothing, location, I...) = true
@inline valid_stress(ϕ::JustRelax.RockRatio, ::Val{:center}, I...) = isvalid_c(ϕ, I...)
@inline valid_stress(ϕ::JustRelax.RockRatio, ::Val{:yz}, I...) = isvalid_yz(ϕ, I...)
@inline valid_stress(ϕ::JustRelax.RockRatio, ::Val{:xz}, I...) = isvalid_xz(ϕ, I...)
@inline valid_stress(ϕ::JustRelax.RockRatio, ::Val{:xy}, I...) = isvalid_xy(ϕ, I...)

@parallel_indices (i, j, k) function update_valid_c_mask!(mask::AbstractArray{T, 3}, ϕ::JustRelax.RockRatio) where {T}
    mask[i, j, k] = isvalid_c(ϕ, i, j, k)
    return nothing
end

function compute_variational_kinematics3D!(stokes, dyrel, rheology, phases, ϕ, _di, ni, dt, do_strain_rate, ΔT, melt_fraction)
    @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di.vertex)
    if do_strain_rate
        @parallel (@idx ni .+ 1) compute_strain_rate!(stokes.∇V, @strain(stokes)..., @velocity(stokes)..., ϕ, _di.center)
    end
    @parallel (@idx ni) variational_RP3D!(stokes.R.RP, stokes.P, stokes.P0, stokes.∇V, stokes.Q, dyrel.ηb, ϕ, rheology, phases.center, ΔT, melt_fraction, dt)
    return nothing
end

@parallel_indices (i, j, k) function variational_RP3D!(RP, P, P0, div, Q, ηb, ϕ, rheology, phase, ΔT, melt_fraction, dt)
    RP[i, j, k] = if isvalid_c(ϕ, i, j, k)
        ϕ.center[i, j, k] * _RP_cell(P[i, j, k], P0[i, j, k], div[i, j, k], Q[i, j, k], ηb[i, j, k], dt, rheology, phase, ΔT, melt_fraction, i, j, k)
    else
        zero(eltype(RP))
    end
    return nothing
end

@parallel_indices (i, j, k) function update_valid_v_masks!(mx, my, mz, ϕ::JustRelax.RockRatio)
    if all((i, j, k) .≤ size(mx))
        mx[i, j, k] = isvalid_vx(ϕ, i + 1, j, k)
    end
    if all((i, j, k) .≤ size(my))
        my[i, j, k] = isvalid_vy(ϕ, i, j + 1, k)
    end
    if all((i, j, k) .≤ size(mz))
        mz[i, j, k] = isvalid_vz(ϕ, i, j, k + 1)
    end
    return nothing
end

@parallel_indices (i, j, k) function project_reduced_state!(P, P0, ΔPψ, λ, Vx, Vy, Vz, ϕ::JustRelax.RockRatio, mx, my, mz)
    if !isvalid_c(ϕ, i, j, k)
        P[i, j, k] = P0[i, j, k] = ΔPψ[i, j, k] = λ[i, j, k] = zero(eltype(P))
    end
    if all((i, j, k) .≤ size(mx)) && !mx[i, j, k]
        Vx[i + 1, j + 1, k + 1] = zero(eltype(Vx))
    end
    if all((i, j, k) .≤ size(my)) && !my[i, j, k]
        Vy[i + 1, j + 1, k + 1] = zero(eltype(Vy))
    end
    if all((i, j, k) .≤ size(mz)) && !mz[i, j, k]
        Vz[i + 1, j + 1, k + 1] = zero(eltype(Vz))
    end
    return nothing
end

@inline weighted3(A, w, i, j, k) = A[i, j, k] * w[i, j, k]

@inline function variational_momentum3D(d, V, P, θ, τ, ρg, ϕ, dc, dv, dt, i, j, k)
    xx, yy, zz, yz, xz, xy = τ
    dx, dy, dz = @dxi(dv, i, j, k)
    if d == 1
        dx = @dx(dc, i)
        return _d_xa(xx, ϕ.center, dx, i, j, k) +
            dy * (weighted3(xy, ϕ.xy, i + 1, j + 1, k) - weighted3(xy, ϕ.xy, i + 1, j, k)) +
            dz * (weighted3(xz, ϕ.xz, i + 1, j, k + 1) - weighted3(xz, ϕ.xz, i + 1, j, k)) -
            variational_pressure_gradient(P, ϕ, Val(1), dx, i, j, k) - _d_xa(θ, ϕ.center, dx, i, j, k) - _av_xa(ρg[1], ϕ.center, i, j, k)
    elseif d == 2
        dy = @dy(dc, j)
        return _d_ya(yy, ϕ.center, dy, i, j, k) +
            dx * (weighted3(xy, ϕ.xy, i + 1, j + 1, k) - weighted3(xy, ϕ.xy, i, j + 1, k)) +
            dz * (weighted3(yz, ϕ.yz, i, j + 1, k + 1) - weighted3(yz, ϕ.yz, i, j + 1, k)) -
            variational_pressure_gradient(P, ϕ, Val(2), dy, i, j, k) - _d_ya(θ, ϕ.center, dy, i, j, k) - _av_ya(ρg[2], ϕ.center, i, j, k)
    else
        dz = @dz(dc, k)
        return _d_za(zz, ϕ.center, dz, i, j, k) +
            dx * (weighted3(xz, ϕ.xz, i + 1, j, k + 1) - weighted3(xz, ϕ.xz, i, j, k + 1)) +
            dy * (weighted3(yz, ϕ.yz, i, j + 1, k + 1) - weighted3(yz, ϕ.yz, i, j, k + 1)) -
            variational_pressure_gradient(P, ϕ, Val(3), dz, i, j, k) - _d_za(θ, ϕ.center, dz, i, j, k) - _av_za(ρg[3], ϕ.center, i, j, k) +
            V[3][i + 1, j + 1, k + 1] * _d_za(ρg[3], ϕ.center, dz, i, j, k) * dt
    end
end

@inline valid_momentum3(ϕ, d, i, j, k) = d == 1 ? isvalid_vx(ϕ, i + 1, j, k) : d == 2 ? isvalid_vy(ϕ, i, j + 1, k) : isvalid_vz(ϕ, i, j, k + 1)

@parallel_indices (i, j, k) function compute_PH_residual_V!(Rx, Ry, Rz, Vx, Vy, Vz, P, θ, xx, yy, zz, yz, xz, xy, gx, gy, gz, ϕ::JustRelax.RockRatio, dc, dv, dt)
    R, V, τ, g = (Rx, Ry, Rz), (Vx, Vy, Vz), (xx, yy, zz, yz, xz, xy), (gx, gy, gz)
    for d in 1:3
        if all((i, j, k) .≤ size(R[d]))
            R[d][i, j, k] = valid_momentum3(ϕ, d, i, j, k) ? variational_momentum3D(d, V, P, θ, τ, g, ϕ, dc, dv, dt, i, j, k) : zero(eltype(Rx))
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_DR_residual_update_V!(Rx, Ry, Rz, Vx, Vy, Vz, ux, uy, uz, P, θ, xx, yy, zz, yz, xz, xy, gx, gy, gz, Dx, Dy, Dz, ax, ay, az, bx, by, bz, tx, ty, tz, ϕ::JustRelax.RockRatio, dc, dv, dt)
    R, V, u = (Rx, Ry, Rz), (Vx, Vy, Vz), (ux, uy, uz)
    D, α, β, t = (Dx, Dy, Dz), (ax, ay, az), (bx, by, bz), (tx, ty, tz)
    τ, g = (xx, yy, zz, yz, xz, xy), (gx, gy, gz)
    for d in 1:3
        if all((i, j, k) .≤ size(R[d]))
            if valid_momentum3(ϕ, d, i, j, k)
                r = variational_momentum3D(d, V, P, θ, τ, g, ϕ, dc, dv, dt, i, j, k) / D[d][i, j, k]
                R[d][i, j, k] = r
                unew, ΔV = damped_update_V(u[d][i, j, k], r, α[d][i, j, k], β[d][i, j, k], t[d][i, j, k])
                u[d][i, j, k] = unew
                V[d][i + 1, j + 1, k + 1] += ΔV
            else
                R[d][i, j, k] = u[d][i, j, k] = V[d][i + 1, j + 1, k + 1] = zero(eltype(Rx))
            end
        end
    end
    return nothing
end

function Gershgorin_variational!(::Val{3}, dyrel, stokes, phases, ϕ, rheology, di, dt, ρg = nothing)
    ni = size(stokes.P)
    @parallel (@idx ni) cld.(ni, (32, 4, 1)) (32, 4, 1) _Gershgorin_Stokes3D_SchurComplement!(
        dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz,
        stokes.viscosity.η, dyrel.γ_eff, di.center, di.vertex,
        phases.center, phases.yz, phases.xz, phases.xy, rheology, dt, ϕ, ρg,
    )
    return nothing
end

# The pressure residual contains ϕ.center, and its gradient multiplies by that
# fraction again. γ_eff is the pressure step length, not the Schur coefficient.
@inline _weighted_penalty(γ, ::Nothing, I...) = γ[I...]
@inline _weighted_penalty(γ, ϕ, I...) = isvalid_c(ϕ, I...) ? γ[I...] * ϕ.center[I...]^2 : zero(eltype(γ))

@inline _weighted_ηve_center(::Nothing, args...) = _ηve_center(args...)
@inline function _weighted_ηve_center(ϕ, η, phase, rheology, dt, I...)
    return isvalid_c(ϕ, I...) ? ϕ.center[I...] * _ηve_center(η, phase, rheology, dt, I...) : zero(eltype(η))
end

@inline _weighted_ηve_yz(::Nothing, args...) = _ηve_yz(args...)
@inline function _weighted_ηve_yz(ϕ, η, phase, rheology, dt, ni, periodic, I...)
    return isvalid_yz(ϕ, I...) ? ϕ.yz[I...] * _ηve_yz(η, phase, rheology, dt, ni, periodic, I...) : zero(eltype(η))
end

@inline _weighted_ηve_xz(::Nothing, args...) = _ηve_xz(args...)
@inline function _weighted_ηve_xz(ϕ, η, phase, rheology, dt, ni, periodic, I...)
    return isvalid_xz(ϕ, I...) ? ϕ.xz[I...] * _ηve_xz(η, phase, rheology, dt, ni, periodic, I...) : zero(eltype(η))
end

@inline _weighted_ηve_xy(::Nothing, args...) = _ηve_xy(args...)
@inline function _weighted_ηve_xy(ϕ, η, phase, rheology, dt, ni, periodic, I...)
    return isvalid_xy(ϕ, I...) ? ϕ.xy[I...] * _ηve_xy(η, phase, rheology, dt, ni, periodic, I...) : zero(eltype(η))
end

@inline function set_variational_preconditioner3D!(D, λ, diagonal, row_sum, i, j, k)
    if diagonal > 0
        D[i, j, k] = diagonal
        λ[i, j, k] = row_sum / diagonal
    elseif iszero(diagonal)
        D[i, j, k] = λ[i, j, k] = one(eltype(D))
    else
        D[i, j, k] = λ[i, j, k] = convert(eltype(D), NaN)
    end
    return nothing
end
