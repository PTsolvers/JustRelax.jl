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

@parallel_indices (I...) function assemble_Rx_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dx, 1) && j ≤ size(dyrel.Dx, 2)
        ni_center = size(dyrel.γ_eff)

        Cxx = zero(eltype(dyrel.Dx))
        for m in 1:5
            jac = ∂Rx∂Vx(dyrel, i, j, m)
            Cxx += abs(jac)
            if m == 3
                dyrel.Dx[i, j] = abs(jac)
            end
        end

        Cxy = zero(eltype(dyrel.Dx))
        for m in 1:4
            jac = ∂Rx∂Vy(dyrel, i, j, m)
            Cxy += abs(jac)
        end

        dyrel.λmaxVx[i, j] = inv(dyrel.Dx[i, j]) * (Cxx + Cxy)
    end

    return nothing
end

@parallel_indices (I...) function assemble_Ry_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dy, 1) && j ≤ size(dyrel.Dy, 2)
        ni_center = size(dyrel.γ_eff)

        Cyx = zero(eltype(dyrel.Dy))
        for m in 1:4
            jac = ∂Ry∂Vx(dyrel, i, j, m)
            Cyx += abs(jac)
        end

        Cyy = zero(eltype(dyrel.Dy))
        for m in 1:5
            jac = ∂Ry∂Vy(dyrel, i, j, m)
            Cyy += abs(jac)
            if m == 3
                dyrel.Dy[i, j] = abs(jac)
            end
        end

        dyrel.λmaxVy[i, j] = inv(dyrel.Dy[i, j]) * (Cyx + Cyy)
    end

    return nothing
end

@inline function ∂Rx∂Vx(dyrel, i, j, m)
    if m == 1
        # ∂Rx[i,j] / ∂Vx[i+1,j] (south)
        return dyrel.∂Rx_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i+1,j] * dyrel.∂εxy_∂Vx[1][i+1,j]
    elseif m == 2
        # ∂Rx[i,j] / ∂Vx[i,j+1] (west)
        return dyrel.∂Rx_∂τxx[1][i,j] * dyrel.∂τc_∂ε[1][i,j] * dyrel.∂εxx_∂Vx[1][i,j] +
               dyrel.∂Rx_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vx[1][i,j])
    elseif m == 3
        # ∂Rx[i,j] / ∂Vx[i+1,j+1] (center)
        return dyrel.∂Rx_∂τxx[2][i,j] * dyrel.∂τc_∂ε[1][i+1,j] * dyrel.∂εxx_∂Vx[1][i+1,j] +
               dyrel.∂Rx_∂τxx[1][i,j] * dyrel.∂τc_∂ε[1][i,j] * dyrel.∂εxx_∂Vx[2][i,j] +
               dyrel.∂Rx_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vx[1][i+1,j+1] +
               dyrel.∂Rx_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i+1,j] * dyrel.∂εxy_∂Vx[2][i+1,j] +
               dyrel.∂Rx_∂P_num[2][i,j] * (-dyrel.γ_eff[i+1,j] * dyrel.∂∇V_∂Vx[1][i+1,j]) +
               dyrel.∂Rx_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vx[2][i,j])
    elseif m == 4
        # ∂Rx[i,j] / ∂Vx[i+2,j+1] (east)
        return dyrel.∂Rx_∂τxx[2][i,j] * dyrel.∂τc_∂ε[1][i+1,j] * dyrel.∂εxx_∂Vx[2][i+1,j] +
               dyrel.∂Rx_∂P_num[2][i,j] * (-dyrel.γ_eff[i+1,j] * dyrel.∂∇V_∂Vx[2][i+1,j])
    else
        # ∂Rx[i,j] / ∂Vx[i+1,j+2] (north)
        return dyrel.∂Rx_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vx[2][i+1,j+1]
    end
end

@inline function ∂Rx∂Vy(dyrel, i, j, m)
    if m == 1
        # ∂Rx[i,j] / ∂Vy[i+1,j] (southwest)
        return dyrel.∂Rx_∂τxx[1][i,j] * dyrel.∂τc_∂ε[1][i,j] * dyrel.∂εxx_∂Vy[1][i,j] +
               dyrel.∂Rx_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i+1,j] * dyrel.∂εxy_∂Vy[1][i+1,j] +
               dyrel.∂Rx_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vy[1][i,j])
    elseif m == 2
        # ∂Rx[i,j] / ∂Vy[i+2,j] (southeast)
        return dyrel.∂Rx_∂τxx[2][i,j] * dyrel.∂τc_∂ε[1][i+1,j] * dyrel.∂εxx_∂Vy[1][i+1,j] +
               dyrel.∂Rx_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i+1,j] * dyrel.∂εxy_∂Vy[2][i+1,j] +
               dyrel.∂Rx_∂P_num[2][i,j] * (-dyrel.γ_eff[i+1,j] * dyrel.∂∇V_∂Vy[1][i+1,j])
    elseif m == 3
        # ∂Rx[i,j] / ∂Vy[i+1,j+1] (northwest)
        return dyrel.∂Rx_∂τxx[1][i,j] * dyrel.∂τc_∂ε[1][i,j] * dyrel.∂εxx_∂Vy[2][i,j] +
               dyrel.∂Rx_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vy[1][i+1,j+1] +
               dyrel.∂Rx_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vy[2][i,j])
    else
        # ∂Rx[i,j] / ∂Vy[i+2,j+1] (northeast)
        return dyrel.∂Rx_∂τxx[2][i,j] * dyrel.∂τc_∂ε[1][i+1,j] * dyrel.∂εxx_∂Vy[2][i+1,j] +
               dyrel.∂Rx_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vy[2][i+1,j+1] +
               dyrel.∂Rx_∂P_num[2][i,j] * (-dyrel.γ_eff[i+1,j] * dyrel.∂∇V_∂Vy[2][i+1,j])
    end
end

@inline function ∂Ry∂Vx(dyrel, i, j, m)
    if m == 1
        # ∂Ry[i,j] / ∂Vx[i,j+1] (southwest)
        return dyrel.∂Ry_∂τyy[1][i,j] * dyrel.∂τc_∂ε[5][i,j] * dyrel.∂εyy_∂Vx[1][i,j] +
               dyrel.∂Ry_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i,j+1] * dyrel.∂εxy_∂Vx[1][i,j+1] +
               dyrel.∂Ry_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vx[1][i,j])
    elseif m == 2
        # ∂Ry[i,j] / ∂Vx[i+1,j+1] (southeast)
        return dyrel.∂Ry_∂τyy[1][i,j] * dyrel.∂τc_∂ε[5][i,j] * dyrel.∂εyy_∂Vx[2][i,j] +
               dyrel.∂Ry_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vx[1][i+1,j+1] +
               dyrel.∂Ry_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vx[2][i,j])
    elseif m == 3
        # ∂Ry[i,j] / ∂Vx[i,j+2] (northwest)
        return dyrel.∂Ry_∂τyy[2][i,j] * dyrel.∂τc_∂ε[5][i,j+1] * dyrel.∂εyy_∂Vx[1][i,j+1] +
               dyrel.∂Ry_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i,j+1] * dyrel.∂εxy_∂Vx[2][i,j+1] +
               dyrel.∂Ry_∂P_num[2][i,j] * (-dyrel.γ_eff[i,j+1] * dyrel.∂∇V_∂Vx[1][i,j+1])
    else
        # ∂Ry[i,j] / ∂Vx[i+1,j+2] (northeast)
        return dyrel.∂Ry_∂τyy[2][i,j] * dyrel.∂τc_∂ε[5][i,j+1] * dyrel.∂εyy_∂Vx[2][i,j+1] +
               dyrel.∂Ry_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vx[2][i+1,j+1] +
               dyrel.∂Ry_∂P_num[2][i,j] * (-dyrel.γ_eff[i,j+1] * dyrel.∂∇V_∂Vx[2][i,j+1])
    end
end

@inline function ∂Ry∂Vy(dyrel, i, j, m)
    if m == 1
        # ∂Ry[i,j] / ∂Vy[i+1,j] (south)
        return dyrel.∂Ry_∂τyy[1][i,j] * dyrel.∂τc_∂ε[5][i,j] * dyrel.∂εyy_∂Vy[1][i,j] +
               dyrel.∂Ry_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vy[1][i,j])
    elseif m == 2
        # ∂Ry[i,j] / ∂Vy[i,j+1] (west)
        return dyrel.∂Ry_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i,j+1] * dyrel.∂εxy_∂Vy[1][i,j+1]
    elseif m == 3
        # ∂Ry[i,j] / ∂Vy[i+1,j+1] (center)
        return dyrel.∂Ry_∂τyy[2][i,j] * dyrel.∂τc_∂ε[5][i,j+1] * dyrel.∂εyy_∂Vy[1][i,j+1] +
               dyrel.∂Ry_∂τyy[1][i,j] * dyrel.∂τc_∂ε[5][i,j] * dyrel.∂εyy_∂Vy[2][i,j] +
               dyrel.∂Ry_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vy[1][i+1,j+1] +
               dyrel.∂Ry_∂τxy[1][i,j] * dyrel.∂τv_∂ε[9][i,j+1] * dyrel.∂εxy_∂Vy[2][i,j+1] +
               dyrel.∂Ry_∂P_num[2][i,j] * (-dyrel.γ_eff[i,j+1] * dyrel.∂∇V_∂Vy[1][i,j+1]) +
               dyrel.∂Ry_∂P_num[1][i,j] * (-dyrel.γ_eff[i,j] * dyrel.∂∇V_∂Vy[2][i,j])
    elseif m == 4
        # ∂Ry[i,j] / ∂Vy[i+2,j+1] (east)
        return dyrel.∂Ry_∂τxy[2][i,j] * dyrel.∂τv_∂ε[9][i+1,j+1] * dyrel.∂εxy_∂Vy[2][i+1,j+1]
    else
        # ∂Ry[i,j] / ∂Vy[i+1,j+2] (north)
        return dyrel.∂Ry_∂τyy[2][i,j] * dyrel.∂τc_∂ε[5][i,j+1] * dyrel.∂εyy_∂Vy[2][i,j+1] +
               dyrel.∂Ry_∂P_num[2][i,j] * (-dyrel.γ_eff[i,j+1] * dyrel.∂∇V_∂Vy[2][i,j+1])
    end
end
