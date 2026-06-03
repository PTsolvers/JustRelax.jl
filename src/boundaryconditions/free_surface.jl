function free_surface_bcs!(
        stokes, bcs::AbstractFlowBoundaryConditions, ηeff, _di_vx, _di_vy, ::Val{2}
    )
    return if bcs.free_surface
        @parallel (@idx (size(stokes.V.Vy, 1) - 2)) FreeSurface_Vy!(
            @velocity(stokes)...,
            stokes.τ.yy,
            ηeff,
            _di_vx,
            _di_vy,
        )
    end
end

function free_surface_bcs!(
        stokes, bcs::AbstractFlowBoundaryConditions, ηeff, di_vx, di_vy, di_vz, ::Val{3}
    )
    return if bcs.free_surface
        @parallel (@idx (size(stokes.V.Vz, 1) - 2, size(stokes.V.Vz, 2) - 2)) FreeSurface_Vz!(
            @velocity(stokes)...,
            stokes.τ.zz,
            ηeff,
            di_vx,
            di_vy,
            di_vz,
        )
    end
end

@parallel_indices (i) function FreeSurface_Vy!(
        Vx::AbstractArray{T, 2},
        Vy::AbstractArray{T, 2},
        τyy::AbstractArray{T, 2},
        ηeff::AbstractArray{T, 2},
        di_vx,
        di_vy,
    ) where {T}

    dx = @dx(di_vx, i)
    dy = @dy(di_vy, size(τyy, 2))
    ∂Vx∂x = (Vx[i + 1, end - 1] - Vx[i, end - 1]) / dx
    ∂Vy∂y = ∂Vx∂x / 2 + 3 * τyy[i, end] / (4 * ηeff[i, end])
    Vy[i + 1, end] = Vy[i + 1, end - 1] + ∂Vy∂y * dy
    return nothing
end

@parallel_indices (i, j) function FreeSurface_Vz!(
        Vx::AbstractArray{T, 3},
        Vy::AbstractArray{T, 3},
        Vz::AbstractArray{T, 3},
        τzz::AbstractArray{T, 3},
        ηeff::AbstractArray{T, 3},
        di_vx,
        di_vy,
        di_vz,
    ) where {T}
    dx = @dx(di_vx, i)
    dy = @dy(di_vy, j)
    dz = @dz(di_vz, size(τzz, 3))
    ∂Vx∂x = (Vx[i + 1, j + 1, end - 1] - Vx[i, j + 1, end - 1]) / dx
    ∂Vy∂y = (Vy[i + 1, j + 1, end - 1] - Vy[i + 1, j, end - 1]) / dy
    ∂Vz∂z = (∂Vx∂x + ∂Vy∂y) / 2 + 3 * τzz[i, j, end] / (4 * ηeff[i, j, end])
    Vz[i + 1, j + 1, end] = Vz[i + 1, j + 1, end - 1] + ∂Vz∂z * dz
    return nothing
end

function free_surface_stress_bcs!(stokes, bcs::AbstractFlowBoundaryConditions, ::Val{2})
    return if bcs.free_surface
        @parallel (@idx size(stokes.P, 1)) free_surface_stress_bcs!(stokes.P, stokes.τ.yy)
    end
end

function free_surface_stress_bcs!(stokes, bcs::AbstractFlowBoundaryConditions, ::Val{3})
    return if bcs.free_surface
        @parallel (@idx (size(stokes.P, 1), size(stokes.P, 2))) free_surface_stress_bcs!(stokes.P, stokes.τ.zz)
    end
end

@parallel_indices (I...) function free_surface_stress_bcs!(P, τn)
    τn[I..., end] = P[I..., end]
    return nothing
end
