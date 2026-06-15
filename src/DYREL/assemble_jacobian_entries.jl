function assemble_jacobian(
    dyrel,
    _di_center,
    _di_vertex,
    _di_vx,
    _di_vy,
    )

    ni = size(dyrel.∂Rx_∂Vx[1])
    @parallel (@idx ni)  assemble_Rx!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)
    return nothing
end

@parallel_indices (I...) function assemble_Rx!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I
    # get local grid spacing
    _dx = @dx(_di_center, i)
    _dy = @dy(_di_vertex, j)

    if i ≤ size(dyrel.∂Rx_∂Vx[1], 1) && j ≤ size(dyrel.∂Rx_∂Vx[1], 2)

        ni_center = size(dyrel.γ_eff)
        for k in 1:9  # 9 velocity points which can influence one local Rx (5 points if no plasticity is active)

            vi, vj = local_Rx_Vx_index(i, j, k)

            # ∂ε/∂Vx
            εW = dε_center_dVx(i,     j, vi, vj, _di_vertex, _di_vx)
            εE = dε_center_dVx(i + 1, j, vi, vj, _di_vertex, _di_vx)
            εS = dε_vertex_dVx(i + 1, j,     vi, vj, _di_vertex, _di_vx, ni_center)
            εN = dε_vertex_dVx(i + 1, j + 1, vi, vj, _di_vertex, _di_vx, ni_center)

            # ∂τ/∂Vx
            dτxxW_dVx = dτ_dV(dyrel.∂τc_∂ε, 1, i,     j, εW.εxx, εW.εyy, εW.εxy)
            dτxxE_dVx = dτ_dV(dyrel.∂τc_∂ε, 1, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
            dτxyS_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j,     εS.εxx, εS.εyy, εS.εxy)
            dτxyN_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εN.εxx, εN.εyy, εN.εxy)

            # ∂ΔPψ/∂Vx
            ΔPψW_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i,     j, εW.εxx, εW.εyy, εW.εxy)
            ΔPψE_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i + 1, j, εE.εxx, εE.εyy, εE.εxy)

            # ∂Pnum/∂Vx
            dPnumW_dVx = dyrel.γ_eff[i,     j] * εW.div
            dPnumE_dVx = dyrel.γ_eff[i + 1, j] * εE.div

            dyrel.∂Rx_∂Vx[k][i, j] =
                _dx * (dτxxE_dVx - dτxxW_dVx) +
                _dy * (dτxyN_dVx - dτxyS_dVx) -
                _dx * (dPnumE_dVx - dPnumW_dVx) -
                _dx * (ΔPψE_dVx - ΔPψW_dVx)
        end
    end

    return nothing
end

@inline function local_Rx_Vx_index(i, j, k)
    ox = (k - 1) % 3
    oy = (k - 1) ÷ 3
    return i + ox, j + oy
end

@inline function dεnormal_center_dVx(ci, cj, vi, vj, _di_vertex)
    dx = @dx(_di_vertex, ci)
    third = one(dx) / 3
    two_thirds = 2 * third

    if vi == ci && vj == cj + 1
        return (εxx = -two_thirds * dx, εyy = third * dx, div = -dx)
    elseif vi == ci + 1 && vj == cj + 1
        return (εxx = two_thirds * dx, εyy = -third * dx, div = dx)
    end
    return (εxx = zero(dx), εyy = zero(dx), div = zero(dx))
end

@inline function dεxy_vertex_dVx(viτ, vjτ, vi, vj, _di_vx)
    dy = @dy(_di_vx, vjτ)
    half = one(dy) / 2

    if vi == viτ && vj == vjτ
        return -half * dy
    elseif vi == viτ && vj == vjτ + 1
        return half * dy
    end
    return zero(dy)
end

@inline function dε_center_dVx(ci, cj, vi, vj, _di_vertex, _di_vx)
    normal = dεnormal_center_dVx(ci, cj, vi, vj, _di_vertex)
    quarter = one(normal.εxx) / 4
    dεxy = quarter * (
        dεxy_vertex_dVx(ci,     cj,     vi, vj, _di_vx) +
        dεxy_vertex_dVx(ci + 1, cj,     vi, vj, _di_vx) +
        dεxy_vertex_dVx(ci,     cj + 1, vi, vj, _di_vx) +
        dεxy_vertex_dVx(ci + 1, cj + 1, vi, vj, _di_vx)
    )

    return (εxx = normal.εxx, εyy = normal.εyy, εxy = dεxy, div = normal.div)
end

@inline function dε_vertex_dVx(viτ, vjτ, vi, vj, _di_vertex, _di_vx, ni_center)
    i0, j0, ic, jc = clamped_indices(ni_center, viτ, vjτ)

    nSW = dεnormal_center_dVx(i0, j0, vi, vj, _di_vertex)
    nSE = dεnormal_center_dVx(ic, j0, vi, vj, _di_vertex)
    nNW = dεnormal_center_dVx(i0, jc, vi, vj, _di_vertex)
    nNE = dεnormal_center_dVx(ic, jc, vi, vj, _di_vertex)

    quarter = one(nSW.εxx) / 4
    dεxx = quarter * (nSW.εxx + nSE.εxx + nNW.εxx + nNE.εxx)
    dεyy = quarter * (nSW.εyy + nSE.εyy + nNW.εyy + nNE.εyy)
    dεxy = dεxy_vertex_dVx(viτ, vjτ, vi, vj, _di_vx)

    return (εxx = dεxx, εyy = dεyy, εxy = dεxy)
end

@inline function dτ_dV(∂τ_∂ε, row, i, j, dεxx_dV, dεyy_dV, dεxy_dV)
    o = 3 * (row - 1)
    return ∂τ_∂ε[o + 1][i, j] * dεxx_dV +
           ∂τ_∂ε[o + 2][i, j] * dεyy_dV +
           ∂τ_∂ε[o + 3][i, j] * dεxy_dV
end

@inline function dΔPψ_dV(∂ΔPψ_∂ε, i, j, dεxx_dV, dεyy_dV, dεxy_dV)
    return ∂ΔPψ_∂ε[1][i, j] * dεxx_dV +
           ∂ΔPψ_∂ε[2][i, j] * dεyy_dV +
           ∂ΔPψ_∂ε[3][i, j] * dεxy_dV
end
