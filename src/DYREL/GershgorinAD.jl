function Gershgorin_Stokes2D_SchurComplementAD(
        dyrel,
        _di_center,
        _di_vertex,
        _di_vx,
        _di_vy,
    )

    ni = size(dyrel.Dx)
    @parallel (@idx ni) assemble_Rx_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    ni = size(dyrel.Dy)
    @parallel (@idx ni) assemble_Ry_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)
    return nothing
end

@inline function Gershgorin_Stokes_SchurComplementAD!(::Val{2}, dyrel, grid)
    return Gershgorin_Stokes2D_SchurComplementAD(dyrel, grid._di.center, grid._di.vertex, grid._di.velocity[1], grid._di.velocity[2])
end

@inline function Gershgorin_Stokes_SchurComplementAD!(::Val{3}, dyrel, grid)
    error("Not yet implemented for 3D")
end

@parallel_indices (I...) function assemble_Rx_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dx, 1) && j ≤ size(dyrel.Dx, 2)
        Cxx = zero(eltype(dyrel.Dx))
        for m in 1:5
            jac = ∂Rx∂Vx(dyrel, _di_center, _di_vertex, _di_vx, i, j, m)
            Cxx += abs(jac)
            if m == 3
                dyrel.Dx[i, j] = abs(jac)
            end
        end

        Cxy = zero(eltype(dyrel.Dx))
        for m in 1:4
            jac = ∂Rx∂Vy(dyrel, _di_center, _di_vertex, _di_vy, i, j, m)
            Cxy += abs(jac)
        end

        dyrel.λmaxVx[i, j] = inv(dyrel.Dx[i, j]) * (Cxx + Cxy)
    end

    return nothing
end

@parallel_indices (I...) function assemble_Ry_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dy, 1) && j ≤ size(dyrel.Dy, 2)
        Cyx = zero(eltype(dyrel.Dy))
        for m in 1:4
            jac = ∂Ry∂Vx(dyrel, _di_center, _di_vertex, _di_vx, i, j, m)
            Cyx += abs(jac)
        end

        Cyy = zero(eltype(dyrel.Dy))
        for m in 1:5
            jac = ∂Ry∂Vy(dyrel, _di_center, _di_vertex, _di_vy, i, j, m)
            Cyy += abs(jac)
            if m == 3
                dyrel.Dy[i, j] = abs(jac)
            end
        end

        dyrel.λmaxVy[i, j] = inv(dyrel.Dy[i, j]) * (Cyx + Cyy)
    end

    return nothing
end

# Calculates (∂εxx/∂Vx, ∂εyy/∂Vx, ∂∇V/∂Vx) for the positive x-side velocity.
@inline function ∂normal_∂Vx(_di_vertex, i, j)
    _dx, _ = @dxi(_di_vertex, i, j)
    return (2 * _dx / 3, -_dx / 3, _dx)
end

# Calculates (∂εxx/∂Vy, ∂εyy/∂Vy, ∂∇V/∂Vy) for the positive y-side velocity.
@inline function ∂normal_∂Vy(_di_vertex, i, j)
    _, _dy = @dxi(_di_vertex, i, j)
    return (-_dy / 3, 2 * _dy / 3, _dy)
end

# Calculates ∂εxy/∂Vx for the positive y-side velocity.
@inline function ∂shear_∂Vx(_di_vx, j)
    _dy = @dy(_di_vx, j)
    return 0.5 * _dy
end

# Calculates ∂εxy/∂Vy for the positive x-side velocity.
@inline function ∂shear_∂Vy(_di_vy, i)
    _dx = @dx(_di_vy, i)
    return 0.5 * _dx
end

# Calculates ∂Rx/∂τxx for the x-normal stress stencil.
@inline ∂Rx_∂τxx(_di_center, i) = @dx(_di_center, i)
# Calculates ∂Rx/∂τxy for the xy-shear stress stencil.
@inline ∂Rx_∂τxy(_di_vertex, j) = @dy(_di_vertex, j)
# Calculates ∂Rx/∂P_num for the pressure-correction stencil.
@inline ∂Rx_∂Pnum(_di_center, i) = @dx(_di_center, i)
# Calculates ∂Ry/∂τyy for the y-normal stress stencil.
@inline ∂Ry_∂τyy(_di_center, j) = @dy(_di_center, j)
# Calculates ∂Ry/∂τxy for the xy-shear stress stencil.
@inline ∂Ry_∂τxy(_di_vertex, i) = @dx(_di_vertex, i)
# Calculates ∂Ry/∂P_num for the pressure-correction stencil.
@inline ∂Ry_∂Pnum(_di_center, j) = @dy(_di_center, j)

# Calculates ∂Rx[i,j]/∂Vx_m for the five-point Vx stencil.
@inline function ∂Rx∂Vx(dyrel, _di_center, _di_vertex, _di_vx, i, j, m)
    dτxx = ∂Rx_∂τxx(_di_center, i)
    dτxy = ∂Rx_∂τxy(_di_vertex, j)
    dPnum = ∂Rx_∂Pnum(_di_center, i)

    if m == 1
        # ∂Rx[i,j] / ∂Vx[i+1,j] (south)
        return dτxy * dyrel.∂τxyv_∂εxy[i + 1, j] * ∂shear_∂Vx(_di_vx, j)
    elseif m == 2
        # ∂Rx[i,j] / ∂Vx[i,j+1] (west)
        dεxx, _, d∇V = ∂normal_∂Vx(_di_vertex, i, j)
        return dτxx * dyrel.∂τxxc_∂εxx[i, j] * dεxx +
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    elseif m == 3
        # ∂Rx[i,j] / ∂Vx[i+1,j+1] (center)
        dεxx_E, _, d∇V_E = ∂normal_∂Vx(_di_vertex, i + 1, j)
        dεxx_W, _, d∇V_W = ∂normal_∂Vx(_di_vertex, i, j)
        return -dτxx * dyrel.∂τxxc_∂εxx[i + 1, j] * dεxx_E -
            dτxx * dyrel.∂τxxc_∂εxx[i, j] * dεxx_W -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vx(_di_vx, j + 1) -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j] * ∂shear_∂Vx(_di_vx, j) -
            dPnum * (dyrel.γ_eff[i + 1, j] * d∇V_E) -
            dPnum * (dyrel.γ_eff[i, j] * d∇V_W)
    elseif m == 4
        # ∂Rx[i,j] / ∂Vx[i+2,j+1] (east)
        dεxx, _, d∇V = ∂normal_∂Vx(_di_vertex, i + 1, j)
        return dτxx * dyrel.∂τxxc_∂εxx[i + 1, j] * dεxx +
            dPnum * (dyrel.γ_eff[i + 1, j] * d∇V)
    else
        # ∂Rx[i,j] / ∂Vx[i+1,j+2] (north)
        return dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vx(_di_vx, j + 1)
    end
end

# Calculates ∂Rx[i,j]/∂Vy_m for the four-point Vy stencil.
@inline function ∂Rx∂Vy(dyrel, _di_center, _di_vertex, _di_vy, i, j, m)
    dτxx = ∂Rx_∂τxx(_di_center, i)
    dτxy = ∂Rx_∂τxy(_di_vertex, j)
    dPnum = ∂Rx_∂Pnum(_di_center, i)

    if m == 1
        # ∂Rx[i,j] / ∂Vy[i+1,j] (southwest)
        dεxx, _, d∇V = ∂normal_∂Vy(_di_vertex, i, j)
        return dτxx * dyrel.∂τxxc_∂εxx[i, j] * dεxx +
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j] * ∂shear_∂Vy(_di_vy, i + 1) +
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    elseif m == 2
        # ∂Rx[i,j] / ∂Vy[i+2,j] (southeast)
        dεxx, _, d∇V = ∂normal_∂Vy(_di_vertex, i + 1, j)
        return -dτxx * dyrel.∂τxxc_∂εxx[i + 1, j] * dεxx -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j] * ∂shear_∂Vy(_di_vy, i + 1) -
            dPnum * (dyrel.γ_eff[i + 1, j] * d∇V)
    elseif m == 3
        # ∂Rx[i,j] / ∂Vy[i+1,j+1] (northwest)
        dεxx, _, d∇V = ∂normal_∂Vy(_di_vertex, i, j)
        return -dτxx * dyrel.∂τxxc_∂εxx[i, j] * dεxx -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vy(_di_vy, i + 1) -
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    else
        # ∂Rx[i,j] / ∂Vy[i+2,j+1] (northeast)
        dεxx, _, d∇V = ∂normal_∂Vy(_di_vertex, i + 1, j)
        return dτxx * dyrel.∂τxxc_∂εxx[i + 1, j] * dεxx +
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vy(_di_vy, i + 1) +
            dPnum * (dyrel.γ_eff[i + 1, j] * d∇V)
    end
end

# Calculates ∂Ry[i,j]/∂Vx_m for the four-point Vx stencil.
@inline function ∂Ry∂Vx(dyrel, _di_center, _di_vertex, _di_vx, i, j, m)
    dτyy = ∂Ry_∂τyy(_di_center, j)
    dτxy = ∂Ry_∂τxy(_di_vertex, i)
    dPnum = ∂Ry_∂Pnum(_di_center, j)

    if m == 1
        # ∂Ry[i,j] / ∂Vx[i,j+1] (southwest)
        _, dεyy, d∇V = ∂normal_∂Vx(_di_vertex, i, j)
        return dτyy * dyrel.∂τyyc_∂εyy[i, j] * dεyy +
            dτxy * dyrel.∂τxyv_∂εxy[i, j + 1] * ∂shear_∂Vx(_di_vx, j + 1) +
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    elseif m == 2
        # ∂Ry[i,j] / ∂Vx[i+1,j+1] (southeast)
        _, dεyy, d∇V = ∂normal_∂Vx(_di_vertex, i, j)
        return -dτyy * dyrel.∂τyyc_∂εyy[i, j] * dεyy -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vx(_di_vx, j + 1) -
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    elseif m == 3
        # ∂Ry[i,j] / ∂Vx[i,j+2] (northwest)
        _, dεyy, d∇V = ∂normal_∂Vx(_di_vertex, i, j + 1)
        return -dτyy * dyrel.∂τyyc_∂εyy[i, j + 1] * dεyy -
            dτxy * dyrel.∂τxyv_∂εxy[i, j + 1] * ∂shear_∂Vx(_di_vx, j + 1) -
            dPnum * (dyrel.γ_eff[i, j + 1] * d∇V)
    else
        # ∂Ry[i,j] / ∂Vx[i+1,j+2] (northeast)
        _, dεyy, d∇V = ∂normal_∂Vx(_di_vertex, i, j + 1)
        return dτyy * dyrel.∂τyyc_∂εyy[i, j + 1] * dεyy +
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vx(_di_vx, j + 1) +
            dPnum * (dyrel.γ_eff[i, j + 1] * d∇V)
    end
end

# Calculates ∂Ry[i,j]/∂Vy_m for the five-point Vy stencil.
@inline function ∂Ry∂Vy(dyrel, _di_center, _di_vertex, _di_vy, i, j, m)
    dτyy = ∂Ry_∂τyy(_di_center, j)
    dτxy = ∂Ry_∂τxy(_di_vertex, i)
    dPnum = ∂Ry_∂Pnum(_di_center, j)

    if m == 1
        # ∂Ry[i,j] / ∂Vy[i+1,j] (south)
        _, dεyy, d∇V = ∂normal_∂Vy(_di_vertex, i, j)
        return dτyy * dyrel.∂τyyc_∂εyy[i, j] * dεyy +
            dPnum * (dyrel.γ_eff[i, j] * d∇V)
    elseif m == 2
        # ∂Ry[i,j] / ∂Vy[i,j+1] (west)
        return dτxy * dyrel.∂τxyv_∂εxy[i, j + 1] * ∂shear_∂Vy(_di_vy, i)
    elseif m == 3
        # ∂Ry[i,j] / ∂Vy[i+1,j+1] (center)
        _, dεyy_N, d∇V_N = ∂normal_∂Vy(_di_vertex, i, j + 1)
        _, dεyy_S, d∇V_S = ∂normal_∂Vy(_di_vertex, i, j)
        return -dτyy * dyrel.∂τyyc_∂εyy[i, j + 1] * dεyy_N -
            dτyy * dyrel.∂τyyc_∂εyy[i, j] * dεyy_S -
            dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vy(_di_vy, i + 1) -
            dτxy * dyrel.∂τxyv_∂εxy[i, j + 1] * ∂shear_∂Vy(_di_vy, i) -
            dPnum * (dyrel.γ_eff[i, j + 1] * d∇V_N) -
            dPnum * (dyrel.γ_eff[i, j] * d∇V_S)
    elseif m == 4
        # ∂Ry[i,j] / ∂Vy[i+2,j+1] (east)
        return dτxy * dyrel.∂τxyv_∂εxy[i + 1, j + 1] * ∂shear_∂Vy(_di_vy, i + 1)
    else
        # ∂Ry[i,j] / ∂Vy[i+1,j+2] (north)
        _, dεyy, d∇V = ∂normal_∂Vy(_di_vertex, i, j + 1)
        return dτyy * dyrel.∂τyyc_∂εyy[i, j + 1] * dεyy +
            dPnum * (dyrel.γ_eff[i, j + 1] * d∇V)
    end
end
