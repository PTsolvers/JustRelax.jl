function free_surface_bcs!(
        stokes, bcs::AbstractFlowBoundaryConditions, η, rheology, phase_ratios, dt, _di_vx, _di_vy
    )
    return if bcs.free_surface
        @parallel (@idx (size(stokes.V.Vy, 1) - 2)) FreeSurface_Vy!(
            @velocity(stokes)...,
            stokes.P,
            stokes.P0,
            stokes.τ_o.yy,
            η,
            rheology,
            phase_ratios.center,
            dt,
            _di_vx,
            _di_vy,
        )
    end
end

function free_surface_bcs!(
        stokes, bcs::AbstractFlowBoundaryConditions, η, rheology, phase_ratios, dt, di
    )
    return if bcs.free_surface
        @parallel (@idx (size(stokes.V.Vz, 1) - 2, size(stokes.V.Vz, 2) - 2)) FreeSurface_Vy!(
            @velocity(stokes)...,
            stokes.P,
            stokes.P0,
            stokes.τ_o.yy,
            η,
            rheology,
            phase_ratios.center,
            dt,
            di,
        )
    end
end

@parallel_indices (i) function FreeSurface_Vy!(
        Vx::AbstractArray{T, 2},
        Vy::AbstractArray{T, 2},
        P::AbstractArray{T, 2},
        P_old::AbstractArray{T, 2},
        τyy_old::AbstractArray{T, 2},
        η::AbstractArray{T, 2},
        rheology,
        phase_ratios,
        dt::T,
        _di_vx,
        _di_vy,
    ) where {T}
    dx = @dx(_di_vx, i)
    dy = @dy(_di_vy, size(P, 2))
    phase = @inbounds phase_ratios[i, end]
    Gdt = fn_ratio(get_shear_modulus, rheology, phase) * dt
    ν = 1.0e-2
    Vy[i + 1, end] =
        ν * (
        Vy[i + 1, end - 1] +
            (3 / 2) *
            (
            P[i, end] / (2.0 * η[i, end]) + #-
                (τyy_old[i, end] + P_old[i, end]) / (2.0 * Gdt) +
                inv(3.0) * (Vx[i + 1, end - 1] - Vx[i, end - 1]) * inv(dx)
        ) *
            dy
    ) + (1 - ν) * Vy[i + 1, end]
    return nothing
end

@parallel_indices (i, j) function FreeSurface_Vy!(
        Vx::AbstractArray{T, 3},
        Vy::AbstractArray{T, 3},
        Vz::AbstractArray{T, 3},
        P::AbstractArray{T, 3},
        P_old::AbstractArray{T, 3},
        τyy_old::AbstractArray{T, 3},
        η::AbstractArray{T, 3},
        rheology,
        phase_ratios,
        dt::T,
        di,
    ) where {T}
    dx, dy, dz = @dxi(di, i, j, size(P, 3))
    phase = @inbounds phase_ratios[i, j, end]
    Gdt = fn_ratio(get_shear_modulus, rheology, phase) * dt
    Vz[i + 1, j + 1, end] =
        Vz[i + 1, j + 1, end - 1] +
        3.0 / 2.0 *
        (
        P[i, j, end] / (2.0 * η[i, j, end]) -
            (τyy_old[i, j, end] + P_old[i, j, end]) / (2.0 * Gdt) +
            inv(3.0) * (
            (Vx[i + 1, j + 1, end - 1] - Vx[i, j + 1, end - 1]) * inv(dx) +
                (Vy[i + 1, j + 1, end - 1] - Vy[i + 1, j, end - 1]) * inv(dy)
        )
    ) *
        dz
    return nothing
end
