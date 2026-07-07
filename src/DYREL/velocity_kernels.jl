# function compute_local_strain_rates!(
#         εxx,
#         εyy,
#         εxy,
#         dyrel,
#         ∇V,
#         Vx,
#         Vy,
#         _di_vertex,
#         _di_vx,
#         _di_vy,
#         do_partials::Bool,
#         i,
#         j,
#     )

#     @inbounds begin
#         vx_s = Vx[i, j]
#         vx_n = Vx[i, j + 1]
#         vy_w = Vy[i, j]
#         vy_e = Vy[i + 1, j]

#         if i ≤ size(εxy, 1) && j ≤ size(εxy, 2)
#             _dy_vx     = @dy(_di_vx, j)
#             _dx_vy     = @dx(_di_vy, i)
#             Vxᵢⱼ_shear = SA[vx_s, vx_n]
#             Vyᵢⱼ_shear = SA[vy_w, vy_e]

#             εxy[i, j] = local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy)

#             if do_partials
#                 ∂εxy_∂Vxᵢⱼ = ForwardDiff.gradient(Vxᵢⱼ_shear -> local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy), Vxᵢⱼ_shear)
#                 ∂εxy_∂Vyᵢⱼ = ForwardDiff.gradient(Vyᵢⱼ_shear -> local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy), Vyᵢⱼ_shear)
#                 dyrel.∂εxy_∂Vx[1][i, j] = ∂εxy_∂Vxᵢⱼ[1]
#                 dyrel.∂εxy_∂Vx[2][i, j] = ∂εxy_∂Vxᵢⱼ[2]
#                 dyrel.∂εxy_∂Vy[1][i, j] = ∂εxy_∂Vyᵢⱼ[1]
#                 dyrel.∂εxy_∂Vy[2][i, j] = ∂εxy_∂Vyᵢⱼ[2]
#             end
#         end

#         if i ≤ size(∇V, 1) && j ≤ size(∇V, 2)
#             vx_ne    = Vx[i + 1, j + 1]
#             vy_ne    = Vy[i + 1, j + 1]
#             _dx, _dy = @dxi(_di_vertex, i, j)
#             Vxᵢⱼ     = SA[vx_n, vx_ne]
#             Vyᵢⱼ     = SA[vy_e, vy_ne]

#             εxx[i, j], εyy[i, j], ∇V[i, j] = local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy)

#             if do_partials
#                 J_normal = ForwardDiff.jacobian(Vxᵢⱼ -> local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy), Vxᵢⱼ)
#                 dyrel.∂εxx_∂Vx[1][i, j] = J_normal[1, 1]
#                 dyrel.∂εxx_∂Vx[2][i, j] = J_normal[1, 2]
#                 dyrel.∂εyy_∂Vx[1][i, j] = J_normal[2, 1]
#                 dyrel.∂εyy_∂Vx[2][i, j] = J_normal[2, 2]
#                 dyrel.∂∇V_∂Vx[1][i, j] = J_normal[3, 1]
#                 dyrel.∂∇V_∂Vx[2][i, j] = J_normal[3, 2]

#                 J_normal = ForwardDiff.jacobian(Vyᵢⱼ -> local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy), Vyᵢⱼ)
#                 dyrel.∂εxx_∂Vy[1][i, j] = J_normal[1, 1]
#                 dyrel.∂εxx_∂Vy[2][i, j] = J_normal[1, 2]
#                 dyrel.∂εyy_∂Vy[1][i, j] = J_normal[2, 1]
#                 dyrel.∂εyy_∂Vy[2][i, j] = J_normal[2, 2]
#                 dyrel.∂∇V_∂Vy[1][i, j] = J_normal[3, 1]
#                 dyrel.∂∇V_∂Vy[2][i, j] = J_normal[3, 2]
#             end
#         end
#     end

#     return nothing
# end

function local_strain_rate_normal_components(Vx, Vy, _dx, _dy)
    dVx_dx = (Vx[2] - Vx[1]) * _dx
    dVy_dy = (Vy[2] - Vy[1]) * _dy
    div_ij = dVx_dx + dVy_dy
    third = typeof(div_ij)(1) / typeof(div_ij)(3)
    div_third = div_ij * third
    εxx = dVx_dx - div_third
    εyy = dVy_dy - div_third
    return SA[εxx, εyy, div_ij]
end
function local_strain_rate_shear_components(Vx, Vy, _dy_vx, _dx_vy)
    dVx_dy = (Vx[2] - Vx[1]) * _dy_vx
    dVy_dx = (Vy[2] - Vy[1]) * _dx_vy
    εxy = 0.5 * (dVx_dy + dVy_dx)
    return εxy
end

## DIVERGENCE + DEVIATORIC STRAIN RATE TENSOR
# function compute_∇V_strain_rate!(stokes, _di, ni, dim::Val{2})
#     @parallel (@idx ni .+ 1) compute_local_strain_rates!(
#         stokes.ε.xx,
#         stokes.ε.yy,
#         stokes.ε.xy,
#         nothing,
#         stokes.∇V,
#         stokes.V.Vx,
#         stokes.V.Vy,
#         _di.vertex,
#         _di.velocity...,
#         false,
#     )
#     return interpolate_shear_ε_to_centers(stokes, dim)
# end

# function compute_∇V_strain_rate!(stokes, _di, ni, dim)
#     @parallel (@idx ni .+ 1) compute_∇V_strain_rate!(
#         stokes.∇V,
#         @strain(stokes)...,
#         @velocity(stokes)...,
#         _di.vertex,
#         _di.velocity...
#     )
#     return interpolate_shear_ε_to_centers(stokes, dim)
# end

function interpolate_shear_ε_to_centers(stokes, ::Val{2})
    vertex2center!(stokes.ε.xy_c, stokes.ε.xy)
    return nothing
end

function interpolate_shear_ε_to_centers(stokes, ::Val{3})
    vertex2center!(stokes.ε.yz_c, stokes.ε.yz)
    vertex2center!(stokes.ε.xz_c, stokes.ε.xz)
    vertex2center!(stokes.ε.xy_c, stokes.ε.xy)
    return nothing
end

# @parallel_indices (i, j) function compute_∇V_strain_rate!(
#         ∇V::AbstractArray{T, 2},
#         εxx::AbstractArray{T, 2},
#         εyy,
#         εxy,
#         Vx,
#         Vy,
#         _di_vertex,
#         _di_vx,
#         _di_vy,
#     ) where {T}

#     third = T(1) / T(3)

#     @inbounds begin
#         vx_s = Vx[i, j]
#         vx_n = Vx[i, j + 1]
#         vy_w = Vy[i, j]
#         vy_e = Vy[i + 1, j]

#         if i ≤ size(εxy, 1) && j ≤ size(εxy, 2)
#             _dy_vx = @dy(_di_vx, j)
#             _dx_vy = @dx(_di_vy, i)

#             dVx_dy = (vx_n - vx_s) * _dy_vx
#             dVy_dx = (vy_e - vy_w) * _dx_vy
#             εxy[i, j] = 0.5 * (dVx_dy + dVy_dx)
#         end

#         if i ≤ size(∇V, 1) && j ≤ size(∇V, 2)
#             vx_ne = Vx[i + 1, j + 1]
#             vy_ne = Vy[i + 1, j + 1]
#             _dx, _dy = @dxi(_di_vertex, i, j)

#             dVx_dx = (vx_ne - vx_n) * _dx
#             dVy_dy = (vy_ne - vy_e) * _dy
#             div_ij = dVx_dx + dVy_dy
#             ∇V[i, j] = div_ij

#             div_third = div_ij * third
#             εxx[i, j] = dVx_dx - div_third
#             εyy[i, j] = dVy_dy - div_third
#         end
#     end

#     return nothing
# end

# @parallel_indices (i, j, k) function compute_∇V_strain_rate!(
#         ∇V::AbstractArray{T, 3},
#         εxx,
#         εyy,
#         εzz,
#         εyz,
#         εxz,
#         εxy,
#         Vx,
#         Vy,
#         Vz,
#         _di_vertex,
#         _di_vx,
#         _di_vy,
#         _di_vz,
#     ) where {T}

#     third = T(1) / T(3)

#     @inbounds begin
#         if all((i, j, k) .≤ size(∇V))
#             _dx, _dy, _dz = @dxi(_di_vertex, i, j, k)
#             dVx_dx = (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) * _dx
#             dVy_dy = (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) * _dy
#             dVz_dz = (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) * _dz
#             div_ijk = dVx_dx + dVy_dy + dVz_dz
#             ∇V[i, j, k] = div_ijk

#             div_third = div_ijk * third
#             εxx[i, j, k] = dVx_dx - div_third
#             εyy[i, j, k] = dVy_dy - div_third
#             εzz[i, j, k] = dVz_dz - div_third
#         end

#         if all((i, j, k) .≤ size(εyz))
#             _dz_vy = @dz(_di_vy, k)
#             _dy_vz = @dy(_di_vz, j)
#             εyz[i, j, k] =
#                 0.5 * (
#                 _dz_vy * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
#                     _dy_vz * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
#             )
#         end

#         if all((i, j, k) .≤ size(εxz))
#             _dz_vx = @dz(_di_vx, k)
#             _dx_vz = @dx(_di_vz, i)
#             εxz[i, j, k] =
#                 0.5 * (
#                 _dz_vx * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
#                     _dx_vz * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
#             )
#         end

#         if all((i, j, k) .≤ size(εxy))
#             _dy_vx = @dy(_di_vx, j)
#             _dx_vy = @dx(_di_vy, i)
#             εxy[i, j, k] =
#                 0.5 * (
#                 _dy_vx * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
#                     _dx_vy * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
#             )
#         end
#     end

#     return nothing
# end

## DIVERGENCE + DEVIATORIC STRAIN RATE + PRESSURE RESIDUAL (fused)
# Same as `compute_∇V_strain_rate!` but additionally evaluates the pressure residual RP in the same
# pass, reusing the in-register divergence `div_ij` instead of reading ∇V back. RP is a local
# per-center write (no halo needed). The numerical pressure P_num = γ_eff·RP is NOT materialized
# here — it is folded with the (similarly small) plastic pressure correction ΔPψ into the single
# `θc` correction array by the stress kernel. Only these two small corrections are summed; the large
# hydrostatic P is kept separate in the momentum kernel to preserve precision (see
# compute_stress_viscosity_DRYEL!).
# NB: ∇V itself is NOT stored here — it is dead inside the DYREL/PH loop (RP is derived from the
# in-register `div_ij`, and nothing on this path reads ∇V back). The public `stokes.∇V` diagnostic
# is recomputed once from the converged velocity field after the loop in `_solve_DYREL!`.

function compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, do_partials::Bool = false)
    return compute_∇V_strain_rate_RP!(
        Val(ndims(stokes.P)),
        stokes,
        dyrel,
        rheology,
        phase_ratios,
        _di,
        ni,
        dt,
        args,
        do_partials,
    )
end

function compute_∇V_strain_rate_RP!(::Val{2}, stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, do_partials::Bool)
    ΔT = haskey(args, :ΔT) ? args.ΔT : nothing
    melt_fraction = haskey(args, :melt_fraction) ? args.melt_fraction : nothing
    @parallel (@idx ni .+ 1) compute_∇V_strain_rate_RP!(
        stokes.ε.xx,
        stokes.ε.yy,
        stokes.ε.xy,
        dyrel,
        @velocity(stokes)...,
        stokes.R.RP,
        stokes.P,
        stokes.P0,
        stokes.Q,
        dyrel.ηb,
        _di.vertex,
        _di.velocity...,
        rheology,
        phase_ratios.center,
        ΔT,
        melt_fraction,
        dt,
        do_partials,
    )
    # NB: no vertex→center shear-strain interpolation here — ε.*_c is not read inside the DYREL
    # loop (stress reads ε.xy at vertices; τII viscosity reads τ.xy_c). The center strain arrays
    # are re-derived once after the loop by shear2center!(stokes.ε).
    return nothing
end

function compute_∇V_strain_rate_RP!(::Val{3}, stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, do_partials::Bool)
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
        _di.vertex,
        _di.velocity...,
        rheology,
        phase_ratios.center,
        ΔT,
        melt_fraction,
        dt,
    )
    # NB: no vertex→center shear-strain interpolation here — ε.*_c is not read inside the DYREL
    # loop (stress reads ε.xy at vertices; τII viscosity reads τ.xy_c). The center strain arrays
    # are re-derived once after the loop by shear2center!(stokes.ε).
    return nothing
end

@parallel_indices (i, j) function compute_∇V_strain_rate_RP!(
        εxx::AbstractArray{T, 2},
        εyy,
        εxy,
        dyrel,
        Vx,
        Vy,
        RP,
        P,
        P0,
        Q,
        ηb,
        _di_vertex,
        _di_vx,
        _di_vy,
        rheology,
        phase_ratio,
        ΔT,
        melt_fraction,
        dt,
        do_partials,
    ) where {T}

    @inbounds begin
        vx_s = Vx[i, j]
        vx_n = Vx[i, j + 1]
        vy_w = Vy[i, j]
        vy_e = Vy[i + 1, j]

        if i ≤ size(εxy, 1) && j ≤ size(εxy, 2)
            _dy_vx     = @dy(_di_vx, j)
            _dx_vy     = @dx(_di_vy, i)
            Vxᵢⱼ_shear = SA[vx_s, vx_n]
            Vyᵢⱼ_shear = SA[vy_w, vy_e]

            εxy[i, j] = local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy)

            if do_partials
                ∂εxy_∂Vxᵢⱼ = ForwardDiff.gradient(Vxᵢⱼ_shear -> local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy), Vxᵢⱼ_shear)
                ∂εxy_∂Vyᵢⱼ = ForwardDiff.gradient(Vyᵢⱼ_shear -> local_strain_rate_shear_components(Vxᵢⱼ_shear, Vyᵢⱼ_shear, _dy_vx, _dx_vy), Vyᵢⱼ_shear)
                dyrel.∂εxy_∂Vx[1][i, j] = ∂εxy_∂Vxᵢⱼ[1]
                dyrel.∂εxy_∂Vx[2][i, j] = ∂εxy_∂Vxᵢⱼ[2]
                dyrel.∂εxy_∂Vy[1][i, j] = ∂εxy_∂Vyᵢⱼ[1]
                dyrel.∂εxy_∂Vy[2][i, j] = ∂εxy_∂Vyᵢⱼ[2]
            end
        end

        if i ≤ size(εxx, 1) && j ≤ size(εxx, 2)
            vx_ne    = Vx[i + 1, j + 1]
            vy_ne    = Vy[i + 1, j + 1]
            _dx, _dy = @dxi(_di_vertex, i, j)
            Vxᵢⱼ     = SA[vx_n, vx_ne]
            Vyᵢⱼ     = SA[vy_e, vy_ne]

            εxx[i, j], εyy[i, j], div_ij = local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy)
            RP[i, j] = _RP_cell(P[i, j], P0[i, j], div_ij, Q[i, j], ηb[i, j], dt, rheology, phase_ratio, ΔT, melt_fraction, i, j)

            if do_partials

                J_normal = ForwardDiff.jacobian(Vxᵢⱼ -> local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy), Vxᵢⱼ)
                dyrel.∂εxx_∂Vx[1][i, j] = J_normal[1, 1]
                dyrel.∂εxx_∂Vx[2][i, j] = J_normal[1, 2]
                dyrel.∂εyy_∂Vx[1][i, j] = J_normal[2, 1]
                dyrel.∂εyy_∂Vx[2][i, j] = J_normal[2, 2]
                dyrel.∂∇V_∂Vx[1][i, j] = J_normal[3, 1]
                dyrel.∂∇V_∂Vx[2][i, j] = J_normal[3, 2]

                J_normal = ForwardDiff.jacobian(Vyᵢⱼ -> local_strain_rate_normal_components(Vxᵢⱼ, Vyᵢⱼ, _dx, _dy), Vyᵢⱼ)
                dyrel.∂εxx_∂Vy[1][i, j] = J_normal[1, 1]
                dyrel.∂εxx_∂Vy[2][i, j] = J_normal[1, 2]
                dyrel.∂εyy_∂Vy[1][i, j] = J_normal[2, 1]
                dyrel.∂εyy_∂Vy[2][i, j] = J_normal[2, 2]
                dyrel.∂∇V_∂Vy[1][i, j] = J_normal[3, 1]
                dyrel.∂∇V_∂Vy[2][i, j] = J_normal[3, 2]
            end
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_∇V_strain_rate_RP!(
        εxx::AbstractArray{T, 3},
        εyy,
        εzz,
        εyz,
        εxz,
        εxy,
        Vx,
        Vy,
        Vz,
        RP,
        P,
        P0,
        Q,
        ηb,
        _di_vertex,
        _di_vx,
        _di_vy,
        _di_vz,
        rheology,
        phase_ratio,
        ΔT,
        melt_fraction,
        dt,
    ) where {T}

    third = T(1) / T(3)

    @inbounds begin
        if all((i, j, k) .≤ size(εxx))
            _dx, _dy, _dz = @dxi(_di_vertex, i, j, k)
            dVx_dx = (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) * _dx
            dVy_dy = (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) * _dy
            dVz_dz = (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) * _dz
            div_ijk = dVx_dx + dVy_dy + dVz_dz

            div_third = div_ijk * third
            εxx[i, j, k] = dVx_dx - div_third
            εyy[i, j, k] = dVy_dy - div_third
            εzz[i, j, k] = dVz_dz - div_third

            # fused pressure residual (reuses `div_ijk` in-register); the numerical pressure
            # P_num = γ_eff·RP is folded into θc (with ΔPψ) downstream by the stress kernel.
            RP[i, j, k] = _RP_cell(P[i, j, k], P0[i, j, k], div_ijk, Q[i, j, k], ηb[i, j, k], dt, rheology, phase_ratio, ΔT, melt_fraction, i, j, k)
        end

        if all((i, j, k) .≤ size(εyz))
            _dz_vy = @dz(_di_vy, k)
            _dy_vz = @dy(_di_vz, j)
            εyz[i, j, k] =
                0.5 * (
                _dz_vy * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy_vz * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
            )
        end

        if all((i, j, k) .≤ size(εxz))
            _dz_vx = @dz(_di_vx, k)
            _dx_vz = @dx(_di_vz, i)
            εxz[i, j, k] =
                0.5 * (
                _dz_vx * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx_vz * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
            )
        end

        if all((i, j, k) .≤ size(εxy))
            _dy_vx = @dy(_di_vx, j)
            _dx_vy = @dx(_di_vy, i)
            εxy[i, j, k] =
                0.5 * (
                _dy_vx * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx_vy * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
            )
        end
    end

    return nothing
end

## RESIDUALS

@inline function local_Rx_residual(τxx, τxy, P, ΔPψ, ρgx, _dx, _dy)
    return (τxx[2] - τxx[1]) * _dx +
           (τxy[2] - τxy[1]) * _dy -
           (P[2] - P[1]) * _dx -
           (ΔPψ[2] - ΔPψ[1]) * _dx -
           0.5 * (ρgx[1] + ρgx[2])
end

@inline function local_Ry_residual(τyy, τxy, P, ΔPψ, ρgy, _dy, _dx)
    return (τyy[2] - τyy[1]) * _dy +
           (τxy[2] - τxy[1]) * _dx -
           (P[2] - P[1]) * _dy -
           (ΔPψ[2] - ΔPψ[1]) * _dy -
           0.5 * (ρgy[1] + ρgy[2])
end

@inline function local_DR_Rx_residual(τxx, τxy, P, P_num, ΔPψ, ρgx, _dx, _dy, D)
    return (local_Rx_residual(τxx, τxy, P, ΔPψ, ρgx, _dx, _dy) - (P_num[2] - P_num[1]) * _dx) / D
end

@inline function local_DR_Ry_residual(τyy, τxy, P, P_num, ΔPψ, ρgy, _dy, _dx, D)
    return (local_Ry_residual(τyy, τxy, P, ΔPψ, ρgy, _dy, _dx) - (P_num[2] - P_num[1]) * _dy) / D
end

@inline local_Rx_residual(q, _dx, _dy) = local_Rx_residual(
    SA[q[1], q[2]], SA[q[3], q[4]], SA[q[5], q[6]], SA[q[7], q[8]], SA[q[9], q[10]], _dx, _dy
)

@inline local_Ry_residual(q, _dy, _dx) = local_Ry_residual(
    SA[q[1], q[2]], SA[q[3], q[4]], SA[q[5], q[6]], SA[q[7], q[8]], SA[q[9], q[10]], _dy, _dx
)

@inline local_DR_Rx_residual(q, _dx, _dy, D) = local_DR_Rx_residual(
    SA[q[1], q[2]],
    SA[q[3], q[4]],
    SA[q[5], q[6]],
    SA[q[7], q[8]],
    SA[q[9], q[10]],
    SA[q[11], q[12]],
    _dx,
    _dy,
    D,
)

@inline local_DR_Ry_residual(q, _dy, _dx, D) = local_DR_Ry_residual(
    SA[q[1], q[2]],
    SA[q[3], q[4]],
    SA[q[5], q[6]],
    SA[q[7], q[8]],
    SA[q[9], q[10]],
    SA[q[11], q[12]],
    _dy,
    _dx,
    D,
)

@inline function local_DR_Rx_residual_partials(τxx, τxy, P, P_num, ΔPψ, ρgx, _dx, _dy, D)
    q = SA[τxx[1], τxx[2], τxy[1], τxy[2], P[1], P[2], P_num[1], P_num[2], ΔPψ[1], ΔPψ[2], ρgx[1], ρgx[2]]
    ∂R = ForwardDiff.gradient(q -> local_DR_Rx_residual(q, _dx, _dy, one(D)), q)
    return (
        τxx   = SA[∂R[1], ∂R[2]],
        τxy   = SA[∂R[3], ∂R[4]],
        P     = SA[∂R[5], ∂R[6]],
        P_num = SA[∂R[7], ∂R[8]],
        ΔPψ   = SA[∂R[9], ∂R[10]],
    )
end

@inline function local_DR_Ry_residual_partials(τyy, τxy, P, P_num, ΔPψ, ρgy, _dy, _dx, D)
    q = SA[τyy[1], τyy[2], τxy[1], τxy[2], P[1], P[2], P_num[1], P_num[2], ΔPψ[1], ΔPψ[2], ρgy[1], ρgy[2]]
    ∂R = ForwardDiff.gradient(q -> local_DR_Ry_residual(q, _dy, _dx, one(D)), q)
    return (
        τyy   = SA[∂R[1], ∂R[2]],
        τxy   = SA[∂R[3], ∂R[4]],
        P     = SA[∂R[5], ∂R[6]],
        P_num = SA[∂R[7], ∂R[8]],
        ΔPψ   = SA[∂R[9], ∂R[10]],
    )
end

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
        _di_center,
        _di_vertex,
    ) where {T}

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        _dx_c = @dx(_di_center, i)
        _dy_v = @dy(_di_vertex, j)
        τxxᵢⱼ = SA[τxx[i, j], τxx[i + 1, j]]
        τxyᵢⱼ = SA[τxy[i + 1, j], τxy[i + 1, j + 1]]
        Pᵢⱼ   = SA[P[i, j], P[i + 1, j]]
        ΔPψᵢⱼ = SA[ΔPψ[i, j], ΔPψ[i + 1, j]]
        ρgxᵢⱼ = SA[ρgx[i, j], ρgx[i + 1, j]]
        Rx[i, j] = local_Rx_residual(
            τxxᵢⱼ,
            τxyᵢⱼ,
            Pᵢⱼ,
            ΔPψᵢⱼ,
            ρgxᵢⱼ,
            _dx_c,
            _dy_v,
        )
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        _dy_c = @dy(_di_center, j)
        _dx_v = @dx(_di_vertex, i)
        τyyᵢⱼ = SA[τyy[i, j], τyy[i, j + 1]]
        τxyᵢⱼ = SA[τxy[i, j + 1], τxy[i + 1, j + 1]]
        Pᵢⱼ = SA[P[i, j], P[i, j + 1]]
        ΔPψᵢⱼ = SA[ΔPψ[i, j], ΔPψ[i, j + 1]]
        ρgyᵢⱼ = SA[ρgy[i, j], ρgy[i, j + 1]]
        Ry[i, j] = local_Ry_residual(
            τyyᵢⱼ,
            τxyᵢⱼ,
            Pᵢⱼ,
            ΔPψᵢⱼ,
            ρgyᵢⱼ,
            _dy_c,
            _dx_v,
        )
    end
    return nothing
end

# @parallel_indices (i, j) function compute_PH_residual_V!(
#         Rx::AbstractArray{T, 2},
#         Ry,
#         Vx,
#         Vy,
#         P,
#         ΔPψ,
#         τxx,
#         τyy,
#         τxy,
#         ρgx,
#         ρgy,
#         _di_center,
#         _di_vertex,
#         dt,
#     ) where {T}

#     nx, ny = size(ρgy)
#     if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
#         _dx_c    = @dx(_di_center, i)
#         _dy_v    = @dy(_di_vertex, j)
#         τxxᵢⱼ    = SA[τxx[i, j], τxx[i + 1, j]]
#         τxyᵢⱼ    = SA[τxy[i + 1, j], τxy[i + 1, j + 1]]
#         Pᵢⱼ      = SA[P[i, j], P[i + 1, j]]
#         ΔPψᵢⱼ    = SA[ΔPψ[i, j], ΔPψ[i + 1, j]]
#         ρgxᵢⱼ    = SA[ρgx[i, j], ρgx[i + 1, j]]
#         Rx[i, j] = local_Rx_residual(
#             τxxᵢⱼ,
#             τxyᵢⱼ,
#             Pᵢⱼ,
#             ΔPψᵢⱼ,
#             ρgxᵢⱼ,
#             _dx_c,
#             _dy_v,
#         )
#     end

#     if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
#         _dy_c = @dy(_di_center, j)
#         _dx_v = @dx(_di_vertex, i)
#         θ = 1.0
#         # Vertical velocity
#         Vyᵢⱼ = Vy[i + 1, j + 1]
#         # Get necessary buoyancy forces
#         j_N = min(j + 1, ny)
#         ρg_S = ρgy[i, j]
#         ρg_N = ρgy[i, j_N]
#         # Spatial derivatives
#         ∂ρg∂y = (ρg_N - ρg_S) * _dy_c
#         # correction term
#         ρg_correction = (Vyᵢⱼ * ∂ρg∂y) * θ * dt
#         τyyᵢⱼ = SA[τyy[i, j], τyy[i, j + 1]]
#         τxyᵢⱼ = SA[τxy[i, j + 1], τxy[i + 1, j + 1]]
#         Pᵢⱼ = SA[P[i, j], P[i, j + 1]]
#         ΔPψᵢⱼ = SA[ΔPψ[i, j], ΔPψ[i, j + 1]]
#         ρgyᵢⱼ = SA[ρgy[i, j], ρgy[i, j + 1]]

#         Ry[i, j] = local_Ry_residual(
#             τyyᵢⱼ,
#             τxyᵢⱼ,
#             Pᵢⱼ,
#             ΔPψᵢⱼ,
#             ρgyᵢⱼ,
#             _dy_c,
#             _dx_v,
#         ) + ρg_correction
#     end

#     return nothing
# end

@parallel_indices (i, j) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 2},
        Ry,
        dyrel,
        P,
        P_num,
        ΔPψ,
        τxx,
        τyy,
        τxy,
        ρgx,
        ρgy,
        Dx,
        Dy,
        _di_center,
        _di_vertex,
        do_partials::Bool,
    ) where {T}

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        _dx_c  = @dx(_di_center, i)
        _dy_v  = @dy(_di_vertex, j)
        τxxᵢⱼ  = SA[τxx[i, j], τxx[i + 1, j]]
        τxyᵢⱼ  = SA[τxy[i + 1, j], τxy[i + 1, j + 1]]
        Pᵢⱼ    = SA[P[i, j], P[i + 1, j]]
        Pnumᵢⱼ = SA[P_num[i, j], P_num[i + 1, j]]
        ΔPψᵢⱼ  = SA[ΔPψ[i, j], ΔPψ[i + 1, j]]
        ρgxᵢⱼ  = SA[ρgx[i, j], ρgx[i + 1, j]]
        Rx[i, j] = local_DR_Rx_residual(
            τxxᵢⱼ,
            τxyᵢⱼ,
            Pᵢⱼ,
            Pnumᵢⱼ,
            ΔPψᵢⱼ,
            ρgxᵢⱼ,
            _dx_c,
            _dy_v,
            Dx[i, j],
        )
        if do_partials
            ∂R = local_DR_Rx_residual_partials(τxxᵢⱼ, τxyᵢⱼ, Pᵢⱼ, Pnumᵢⱼ, ΔPψᵢⱼ, ρgxᵢⱼ, _dx_c, _dy_v, Dx[i, j])
            dyrel.∂Rx_∂τxx[1][i, j]   = ∂R.τxx[1]
            dyrel.∂Rx_∂τxx[2][i, j]   = ∂R.τxx[2]
            dyrel.∂Rx_∂τxy[1][i, j]   = ∂R.τxy[1]
            dyrel.∂Rx_∂τxy[2][i, j]   = ∂R.τxy[2]
            dyrel.∂Rx_∂P[1][i, j]     = ∂R.P[1]
            dyrel.∂Rx_∂P[2][i, j]     = ∂R.P[2]
            dyrel.∂Rx_∂P_num[1][i, j] = ∂R.P_num[1]
            dyrel.∂Rx_∂P_num[2][i, j] = ∂R.P_num[2]
        end
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        _dy_c    = @dy(_di_center, j)
        _dx_v    = @dx(_di_vertex, i)
        τyyᵢⱼ    = SA[τyy[i, j], τyy[i, j + 1]]
        τxyᵢⱼ    = SA[τxy[i, j + 1], τxy[i + 1, j + 1]]
        Pᵢⱼ      = SA[P[i, j], P[i, j + 1]]
        Pnumᵢⱼ   = SA[P_num[i, j], P_num[i, j + 1]]
        ΔPψᵢⱼ    = SA[ΔPψ[i, j], ΔPψ[i, j + 1]]
        ρgyᵢⱼ    = SA[ρgy[i, j], ρgy[i, j + 1]]
        Ry[i, j] = local_DR_Ry_residual(
            τyyᵢⱼ,
            τxyᵢⱼ,
            Pᵢⱼ,
            Pnumᵢⱼ,
            ΔPψᵢⱼ,
            ρgyᵢⱼ,
            _dy_c,
            _dx_v,
            Dy[i, j],
        )
        if do_partials
            ∂R = local_DR_Ry_residual_partials(τyyᵢⱼ, τxyᵢⱼ, Pᵢⱼ, Pnumᵢⱼ, ΔPψᵢⱼ, ρgyᵢⱼ, _dy_c, _dx_v, Dy[i, j])
            dyrel.∂Ry_∂τyy[1][i, j]   = ∂R.τyy[1]
            dyrel.∂Ry_∂τyy[2][i, j]   = ∂R.τyy[2]
            dyrel.∂Ry_∂τxy[1][i, j]   = ∂R.τxy[1]
            dyrel.∂Ry_∂τxy[2][i, j]   = ∂R.τxy[2]
            dyrel.∂Ry_∂P[1][i, j]     = ∂R.P[1]
            dyrel.∂Ry_∂P[2][i, j]     = ∂R.P[2]
            dyrel.∂Ry_∂P_num[1][i, j] = ∂R.P_num[1]
            dyrel.∂Ry_∂P_num[2][i, j] = ∂R.P_num[2]
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _di_center, _di_vertex
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)

    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        _dx = @dx(_di_center, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_vertex, k)

        Rx[i, j, k] = d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) - d_xa(P, _dx) - d_xa(ΔPψ, _dx) - av_x(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dz(_di_vertex, k)

        Ry[i, j, k] = d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) - d_ya(P, _dy) - d_ya(ΔPψ, _dy) - av_y(ρgy)
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_center, k)

        Rz[i, j, k] = d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) - d_za(P, _dz) - d_za(ΔPψ, _dz) - av_z(ρgz)
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_PH_residual_V!(
        Rx::AbstractArray{T, 3}, Ry, Rz, Vx, Vy, Vz, P, ΔPψ, τxx, τyy, τzz, τxy, τxz, τyz, ρgx, ρgy, ρgz, _di_center, _di_vertex, dt
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)

    nx, ny, nz = size(ρgz)
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        _dx = @dx(_di_center, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_vertex, k)

        Rx[i, j, k] = d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) - d_xa(P, _dx) - d_xa(ΔPψ, _dx) - av_x(ρgx)
    end

    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dz(_di_vertex, k)

        Ry[i, j, k] = d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) - d_ya(P, _dy) - d_ya(ΔPψ, _dy) - av_y(ρgy)
    end

    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_center, k)

        θ = 1.0
        # Vertical velocity
        Vzᵢⱼₖ = Vz[i + 1, j + 1, k + 1]
        # Get necessary buoyancy forces
        k_T = min(k + 1, nz)
        ρg_B = ρgz[i, j, k]
        ρg_T = ρgz[i, j, k_T]
        # Spatial derivatives
        ∂ρg∂z = (ρg_T - ρg_B) * _dz
        # correction term
        ρg_correction = (Vzᵢⱼₖ * ∂ρg∂z) * θ * dt

        Rz[i, j, k] = d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) - d_za(P, _dz) - d_za(ΔPψ, _dz) - av_z(ρgz) + ρg_correction
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_DR_residual_V!(
        Rx::AbstractArray{T, 3},
        Ry,
        Rz,
        P,
        P_num,
        ΔPψ,
        τxx,
        τyy,
        τzz,
        τxy,
        τxz,
        τyz,
        ρgx,
        ρgy,
        ρgz,
        Dx,
        Dy,
        Dz,
        _di_center,
        _di_vertex,
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)

    # @inbounds begin
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
        _dx = @dx(_di_center, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_vertex, k)

        Rx[i, j, k] =
            (
            d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) -
                d_xa(P, _dx) - d_xa(P_num, _dx) - d_xa(ΔPψ, _dx) - av_x(ρgx)
        ) / Dx[i, j, k]
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_center, j)
        _dz = @dz(_di_vertex, k)

        Ry[i, j, k] =
            (
            d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) -
                d_ya(P, _dy) - d_ya(P_num, _dy) - d_ya(ΔPψ, _dy) - av_y(ρgy)
        ) / Dy[i, j, k]
    end
    if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
        _dx = @dx(_di_vertex, i)
        _dy = @dy(_di_vertex, j)
        _dz = @dz(_di_center, k)

        Rz[i, j, k] =
            (
            d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) -
                d_za(P, _dz) - d_za(P_num, _dz) - d_za(ΔPψ, _dz) - av_z(ρgz)
        ) / Dz[i, j, k]
    end
    # end

    return nothing
end

@parallel_indices (I...) function update_V_damping_DR_V!(
        V::NTuple{N, AbstractArray{T, N}},
        dVdτ::NTuple{N, AbstractArray{T, N}},
        R::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(R[d]))
            dVdτ[d][I...] = αV[d][I...] * dVdτ[d][I...] + R[d][I...]
            V[d][I .+ 1...] += dVdτ[d][I...] * βV[d][I...] * dτV[d][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function compute_dV!(
        dV::NTuple{N, AbstractArray{T, N}},
        dVdτ::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(dV[d]))
            dV[d][I...] = dVdτ[d][I...] * βV[d][I...] * dτV[d][I...]
        end
    end

    return nothing
end

@parallel_indices (I...) function update_cV!(
        cV::NTuple{N, AbstractArray{T, N}}, cV_I
    ) where {N, T}

    ntuple(Val(N)) do d
        @inline
        if all(I .≤ size(cV[d]))
            cV[d][I...] = cV_I
        end
    end

    return nothing
end

## DR VELOCITY RESIDUAL + DAMPED UPDATE (fused)
# Per-index damped pseudo-transient velocity update, shared between the fused residual kernel
# below and the standalone `update_V_damping_DR_V!`. Returns (dVdτⁿ⁺¹, ΔV) so the caller can
# update V without re-reading the residual from global memory. GPU-safe (tuple return).
@inline function damped_update_V(dVdτ, R, α, β, dτ)
    dVdτ_new = α * dVdτ + R
    return dVdτ_new, dVdτ_new * β * dτ
end

# Fuses `compute_DR_residual_V!` (velocity residual R = ∂ⱼτiⱼ − ∂ᵢ(P + θc) − ρgᵢ, /Dᵢ, where the
# small pressure correction θc = P_num + ΔPψ is assembled once per iteration by the stress kernel)
# with the damped update of `update_V_damping_DR_V!`. Folding only the two small corrections (not the
# large hydrostatic P) collapses three neighbour-stencil reads into two while keeping P differenced
# at full precision. R[I] is written to global memory (needed by the residual norm / λmin) and
# immediately reused in-register for the velocity update.
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
        _di_center,
        _di_vertex,
        dyrel,
        do_partials::Bool,
    ) where {T}
    @inbounds begin
        ΔPψᵢⱼ = SA[zero(T), zero(T)]

        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
            _dx_c  = @dx(_di_center, i)
            _dy_v  = @dy(_di_vertex, j)
            τxxᵢⱼ  = SA[τxx[i, j], τxx[i + 1, j]]
            τxyᵢⱼ  = SA[τxy[i + 1, j], τxy[i + 1, j + 1]]
            Pᵢⱼ    = SA[P[i, j], P[i + 1, j]]
            θcᵢⱼ   = SA[θc[i, j], θc[i + 1, j]]
            ρgxᵢⱼ  = SA[ρgx[i, j], ρgx[i + 1, j]]
            Rx_ij  = local_DR_Rx_residual(τxxᵢⱼ, τxyᵢⱼ, Pᵢⱼ, θcᵢⱼ, ΔPψᵢⱼ, ρgxᵢⱼ, _dx_c, _dy_v, Dx[i, j])
            Rx[i, j] = Rx_ij
            if do_partials
                ∂R = local_DR_Rx_residual_partials(τxxᵢⱼ, τxyᵢⱼ, Pᵢⱼ, θcᵢⱼ, ΔPψᵢⱼ, ρgxᵢⱼ, _dx_c, _dy_v, Dx[i, j])
                dyrel.∂Rx_∂τxx[1][i, j]   = ∂R.τxx[1]
                dyrel.∂Rx_∂τxx[2][i, j]   = ∂R.τxx[2]
                dyrel.∂Rx_∂τxy[1][i, j]   = ∂R.τxy[1]
                dyrel.∂Rx_∂τxy[2][i, j]   = ∂R.τxy[2]
                dyrel.∂Rx_∂P[1][i, j]     = ∂R.P[1]
                dyrel.∂Rx_∂P[2][i, j]     = ∂R.P[2]
                dyrel.∂Rx_∂P_num[1][i, j] = ∂R.P_num[1]
                dyrel.∂Rx_∂P_num[2][i, j] = ∂R.P_num[2]
            end

            dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij, αVx[i, j], βVx[i, j], dτVx[i, j])
            dVxdτ[i, j] = dVx_new
            Vx[i + 1, j + 1] += ΔVx
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
            _dy_c  = @dy(_di_center, j)
            _dx_v  = @dx(_di_vertex, i)
            τyyᵢⱼ  = SA[τyy[i, j], τyy[i, j + 1]]
            τxyᵢⱼ  = SA[τxy[i, j + 1], τxy[i + 1, j + 1]]
            Pᵢⱼ    = SA[P[i, j], P[i, j + 1]]
            θcᵢⱼ   = SA[θc[i, j], θc[i, j + 1]]
            ρgyᵢⱼ  = SA[ρgy[i, j], ρgy[i, j + 1]]
            Ry_ij  = local_DR_Ry_residual(τyyᵢⱼ, τxyᵢⱼ, Pᵢⱼ, θcᵢⱼ, ΔPψᵢⱼ, ρgyᵢⱼ, _dy_c, _dx_v, Dy[i, j])
            Ry[i, j] = Ry_ij
            if do_partials
                ∂R = local_DR_Ry_residual_partials(τyyᵢⱼ, τxyᵢⱼ, Pᵢⱼ, θcᵢⱼ, ΔPψᵢⱼ, ρgyᵢⱼ, _dy_c, _dx_v, Dy[i, j])
                dyrel.∂Ry_∂τyy[1][i, j]   = ∂R.τyy[1]
                dyrel.∂Ry_∂τyy[2][i, j]   = ∂R.τyy[2]
                dyrel.∂Ry_∂τxy[1][i, j]   = ∂R.τxy[1]
                dyrel.∂Ry_∂τxy[2][i, j]   = ∂R.τxy[2]
                dyrel.∂Ry_∂P[1][i, j]     = ∂R.P[1]
                dyrel.∂Ry_∂P[2][i, j]     = ∂R.P[2]
                dyrel.∂Ry_∂P_num[1][i, j] = ∂R.P_num[1]
                dyrel.∂Ry_∂P_num[2][i, j] = ∂R.P_num[2]
            end

            dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij, αVy[i, j], βVy[i, j], dτVy[i, j])
            dVydτ[i, j] = dVy_new
            Vy[i + 1, j + 1] += ΔVy
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_DR_residual_update_V!(
        Rx::AbstractArray{T, 3},
        Ry,
        Rz,
        Vx,
        Vy,
        Vz,
        dVxdτ,
        dVydτ,
        dVzdτ,
        P,
        θc,
        τxx,
        τyy,
        τzz,
        τxy,
        τxz,
        τyz,
        ρgx,
        ρgy,
        ρgz,
        Dx,
        Dy,
        Dz,
        αVx,
        αVy,
        αVz,
        βVx,
        βVy,
        βVz,
        dτVx,
        dτVy,
        dτVz,
        _di_center,
        _di_vertex,
        dyrel,
        do_partials::Bool,
    ) where {T}

    Base.@propagate_inbounds @inline d_xa(A, _dx) = _d_xa(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_ya(A, _dy) = _d_ya(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_za(A, _dz) = _d_za(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline d_xi(A, _dx) = _d_xi(A, _dx, i, j, k)
    Base.@propagate_inbounds @inline d_yi(A, _dy) = _d_yi(A, _dy, i, j, k)
    Base.@propagate_inbounds @inline d_zi(A, _dz) = _d_zi(A, _dz, i, j, k)
    Base.@propagate_inbounds @inline av_x(A) = _av_x(A, i, j, k)
    Base.@propagate_inbounds @inline av_y(A) = _av_y(A, i, j, k)
    Base.@propagate_inbounds @inline av_z(A) = _av_z(A, i, j, k)

    @inbounds begin
        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2) && k ≤ size(Rx, 3)
            _dx = @dx(_di_center, i)
            _dy = @dy(_di_vertex, j)
            _dz = @dz(_di_vertex, k)

            Rx_ijk =
                (
                d_xa(τxx, _dx) + d_yi(τxy, _dy) + d_zi(τxz, _dz) -
                    d_xa(P, _dx) - d_xa(θc, _dx) - av_x(ρgx)
            ) / Dx[i, j, k]
            Rx[i, j, k] = Rx_ijk

            dVx_new, ΔVx = damped_update_V(dVxdτ[i, j, k], Rx_ijk, αVx[i, j, k], βVx[i, j, k], dτVx[i, j, k])
            dVxdτ[i, j, k] = dVx_new
            Vx[i + 1, j + 1, k + 1] += ΔVx
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2) && k ≤ size(Ry, 3)
            _dx = @dx(_di_vertex, i)
            _dy = @dy(_di_center, j)
            _dz = @dz(_di_vertex, k)

            Ry_ijk =
                (
                d_ya(τyy, _dy) + d_xi(τxy, _dx) + d_zi(τyz, _dz) -
                    d_ya(P, _dy) - d_ya(θc, _dy) - av_y(ρgy)
            ) / Dy[i, j, k]
            Ry[i, j, k] = Ry_ijk

            dVy_new, ΔVy = damped_update_V(dVydτ[i, j, k], Ry_ijk, αVy[i, j, k], βVy[i, j, k], dτVy[i, j, k])
            dVydτ[i, j, k] = dVy_new
            Vy[i + 1, j + 1, k + 1] += ΔVy
        end
        if i ≤ size(Rz, 1) && j ≤ size(Rz, 2) && k ≤ size(Rz, 3)
            _dx = @dx(_di_vertex, i)
            _dy = @dy(_di_vertex, j)
            _dz = @dz(_di_center, k)

            Rz_ijk =
                (
                d_za(τzz, _dz) + d_xi(τxz, _dx) + d_yi(τyz, _dy) -
                    d_za(P, _dz) - d_za(θc, _dz) - av_z(ρgz)
            ) / Dz[i, j, k]
            Rz[i, j, k] = Rz_ijk

            dVz_new, ΔVz = damped_update_V(dVzdτ[i, j, k], Rz_ijk, αVz[i, j, k], βVz[i, j, k], dτVz[i, j, k])
            dVzdτ[i, j, k] = dVz_new
            Vz[i + 1, j + 1, k + 1] += ΔVz
        end
    end

    return nothing
end
