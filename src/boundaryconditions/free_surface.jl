function free_surface_bcs!(
    stokes, bcs::FlowBoundaryConditions, η, rheology, phase_ratios, dt, di
)
    indices_range(::Any, Vy) = @idx (size(Vy, 1) - 2)
    indices_range(::Any, ::Any, Vz) = @idx (size(Vz, 1) - 2, size(Vz, 2) - 2)

    V = @velocity(stokes)
    n = indices_range(V...)

    if bcs.free_surface
        # apply boundary conditions
        @parallel n FreeSurface_Vy!(
            V...,
            stokes.P,
            stokes.P0,
            stokes.τ_o.yy,
            η,
            rheology,
            phase_ratios.center,
            dt,
            di...,
        )
    end
end

@parallel_indices (i) function FreeSurface_Vy!(
    Vx::AbstractArray{T,2},
    Vy::AbstractArray{T,2},
    P::AbstractArray{T,2},
    P_old::AbstractArray{T,2},
    τyy_old::AbstractArray{T,2},
    η::AbstractArray{T,2},
    rheology,
    phase_ratios,
    dt::T,
    dx::T,
    dy::T,
) where {T}
    phase = @inbounds phase_ratios[i, end]
    Gdt = fn_ratio(get_shear_modulus, rheology, phase) * dt
    ν = 1e-2
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
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    P::AbstractArray{T,3},
    P_old::AbstractArray{T,3},
    τyy_old::AbstractArray{T,3},
    η::AbstractArray{T,3},
    rheology,
    phase_ratios,
    dt::T,
    dx::T,
    dy::T,
    dz::T,
) where {T}
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
