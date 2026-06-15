function assemble_jacobian(
    dyrel,
    _di_center,
    _di_vertex,
    _di_vx,
    _di_vy,
    )

    ni = size(dyrel.∂Rx_∂Vx[1])
    @parallel (@idx ni)  assemble_Rx!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    ni = size(dyrel.∂Ry_∂Vy[1])
    @parallel (@idx ni)  assemble_Ry!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)
    return nothing
end

# Local velocity stencils for one Rx[i,j] residual.
#
# The x=... and y=... labels below are velocity-array indices, not
# physical coordinates. For example, y=j+1 means the row Vx[:, j+1]
# or Vy[:, j+1], depending on the stencil.
#
# Vx stencil, stored in ∂Rx_∂Vx[1:9][i,j]:
#
#   k=7 ---- k=8 ---- k=9      y = j+2
#    |        |        |
#   k=4 ---- k=5 ---- k=6      y = j+1     Rx[i,j] is at k=5
#    |        |        |
#   k=1 ---- k=2 ---- k=3      y = j
#   x=i     x=i+1    x=i+2
#
# Vy stencil, stored in ∂Rx_∂Vy[1:12][i,j]:
#
#   k=9 ---- k=10 --- k=11 --- k=12     y = j+2
#    |        |        |        |
#   k=5 ---- k=6  --- k=7  --- k=8      y = j+1     Rx[i,j] lies between k=6 and k=7
#    |        |        |        |
#   k=1 ---- k=2  --- k=3  --- k=4      y = j
#   x=i     x=i+1    x=i+2    x=i+3

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

        for k in 1:12  # 12 Vy points which can influence one local Rx
            vi, vj = local_Rx_Vy_index(i, j, k)

            # ∂ε/∂Vy
            εW = dε_center_dVy(i,     j, vi, vj, _di_vertex, _di_vy)
            εE = dε_center_dVy(i + 1, j, vi, vj, _di_vertex, _di_vy)
            εS = dε_vertex_dVy(i + 1, j,     vi, vj, _di_vertex, _di_vy, ni_center)
            εN = dε_vertex_dVy(i + 1, j + 1, vi, vj, _di_vertex, _di_vy, ni_center)

            # ∂τ/∂Vy
            dτxxW_dVy = dτ_dV(dyrel.∂τc_∂ε, 1, i,     j, εW.εxx, εW.εyy, εW.εxy)
            dτxxE_dVy = dτ_dV(dyrel.∂τc_∂ε, 1, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
            dτxyS_dVy = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j,     εS.εxx, εS.εyy, εS.εxy)
            dτxyN_dVy = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εN.εxx, εN.εyy, εN.εxy)

            # ∂ΔPψ/∂Vy
            ΔPψW_dVy = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i,     j, εW.εxx, εW.εyy, εW.εxy)
            ΔPψE_dVy = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i + 1, j, εE.εxx, εE.εyy, εE.εxy)

            # ∂Pnum/∂Vy
            dPnumW_dVy = dyrel.γ_eff[i,     j] * εW.div
            dPnumE_dVy = dyrel.γ_eff[i + 1, j] * εE.div

            dyrel.∂Rx_∂Vy[k][i, j] =
                _dx * (dτxxE_dVy - dτxxW_dVy) +
                _dy * (dτxyN_dVy - dτxyS_dVy) -
                _dx * (dPnumE_dVy - dPnumW_dVy) -
                _dx * (ΔPψE_dVy - ΔPψW_dVy)
        end
    end

    return nothing
end

# Local velocity stencils for one Ry[i,j] residual.
#
# The x=... and y=... labels below are velocity-array indices, not
# physical coordinates. For example, y=j+1 means the row Vx[:, j+1]
# or Vy[:, j+1], depending on the stencil.
#
# Vx stencil, stored in ∂Ry_∂Vx[1:12][i,j]:
#
#   k=10 --- k=11 --- k=12     y = j+3
#    |        |        |
#   k=7  --- k=8  --- k=9      y = j+2
#    |        |        |
#   k=4  --- k=5  --- k=6      y = j+1     Ry[i,j] lies between k=5 and k=8
#    |        |        |
#   k=1  --- k=2  --- k=3      y = j
#   x=i     x=i+1    x=i+2
#
# Vy stencil, stored in ∂Ry_∂Vy[1:9][i,j]:
#
#   k=7 ---- k=8 ---- k=9      y = j+2
#    |        |        |
#   k=4 ---- k=5 ---- k=6      y = j+1     Ry[i,j] is at k=5
#    |        |        |
#   k=1 ---- k=2 ---- k=3      y = j
#   x=i     x=i+1    x=i+2
#
@parallel_indices (I...) function assemble_Ry!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I
    # get local grid spacing
    _dy = @dy(_di_center, j)
    _dx = @dx(_di_vertex, i)

    if i ≤ size(dyrel.∂Ry_∂Vy[1], 1) && j ≤ size(dyrel.∂Ry_∂Vy[1], 2)

        ni_center = size(dyrel.γ_eff)
        for k in 1:12  # 12 Vx points which can influence one local Ry

            vi, vj = local_Ry_Vx_index(i, j, k)

            # ∂ε/∂Vx
            εS = dε_center_dVx(i, j,     vi, vj, _di_vertex, _di_vx)
            εN = dε_center_dVx(i, j + 1, vi, vj, _di_vertex, _di_vx)
            εW = dε_vertex_dVx(i,     j + 1, vi, vj, _di_vertex, _di_vx, ni_center)
            εE = dε_vertex_dVx(i + 1, j + 1, vi, vj, _di_vertex, _di_vx, ni_center)

            # ∂τ/∂Vx
            dτyyS_dVx = dτ_dV(dyrel.∂τc_∂ε, 2, i, j,     εS.εxx, εS.εyy, εS.εxy)
            dτyyN_dVx = dτ_dV(dyrel.∂τc_∂ε, 2, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
            dτxyW_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i,     j + 1, εW.εxx, εW.εyy, εW.εxy)
            dτxyE_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εE.εxx, εE.εyy, εE.εxy)

            # ∂ΔPψ/∂Vx
            ΔPψS_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j,     εS.εxx, εS.εyy, εS.εxy)
            ΔPψN_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j + 1, εN.εxx, εN.εyy, εN.εxy)

            # ∂Pnum/∂Vx
            dPnumS_dVx = dyrel.γ_eff[i, j]     * εS.div
            dPnumN_dVx = dyrel.γ_eff[i, j + 1] * εN.div

            dyrel.∂Ry_∂Vx[k][i, j] =
                _dy * (dτyyN_dVx - dτyyS_dVx) +
                _dx * (dτxyE_dVx - dτxyW_dVx) -
                _dy * (dPnumN_dVx - dPnumS_dVx) -
                _dy * (ΔPψN_dVx - ΔPψS_dVx)
        end

        for k in 1:9  # 9 Vy points which can influence one local Ry

            vi, vj = local_Ry_Vy_index(i, j, k)

            # ∂ε/∂Vy
            εS = dε_center_dVy(i, j,     vi, vj, _di_vertex, _di_vy)
            εN = dε_center_dVy(i, j + 1, vi, vj, _di_vertex, _di_vy)
            εW = dε_vertex_dVy(i,     j + 1, vi, vj, _di_vertex, _di_vy, ni_center)
            εE = dε_vertex_dVy(i + 1, j + 1, vi, vj, _di_vertex, _di_vy, ni_center)

            # ∂τ/∂Vy
            dτyyS_dVy = dτ_dV(dyrel.∂τc_∂ε, 2, i, j,     εS.εxx, εS.εyy, εS.εxy)
            dτyyN_dVy = dτ_dV(dyrel.∂τc_∂ε, 2, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
            dτxyW_dVy = dτ_dV(dyrel.∂τv_∂ε, 3, i,     j + 1, εW.εxx, εW.εyy, εW.εxy)
            dτxyE_dVy = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εE.εxx, εE.εyy, εE.εxy)

            # ∂ΔPψ/∂Vy
            ΔPψS_dVy = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j,     εS.εxx, εS.εyy, εS.εxy)
            ΔPψN_dVy = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j + 1, εN.εxx, εN.εyy, εN.εxy)

            # ∂Pnum/∂Vy
            dPnumS_dVy = dyrel.γ_eff[i, j]     * εS.div
            dPnumN_dVy = dyrel.γ_eff[i, j + 1] * εN.div

            dyrel.∂Ry_∂Vy[k][i, j] =
                _dy * (dτyyN_dVy - dτyyS_dVy) +
                _dx * (dτxyE_dVy - dτxyW_dVy) -
                _dy * (dPnumN_dVy - dPnumS_dVy) -
                _dy * (ΔPψN_dVy - ΔPψS_dVy)
        end
    end

    return nothing
end

@inline function local_Rx_Vx_index(i, j, k)
    ox = (k - 1) % 3
    oy = (k - 1) ÷ 3
    return i + ox, j + oy
end

@inline function local_Rx_Vy_index(i, j, k)
    ox = (k - 1) % 4
    oy = (k - 1) ÷ 4
    return i + ox, j + oy
end

@inline function local_Ry_Vx_index(i, j, k)
    ox = (k - 1) % 3
    oy = (k - 1) ÷ 3
    return i + ox, j + oy
end

@inline function local_Ry_Vy_index(i, j, k)
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

@inline function dεnormal_center_dVy(ci, cj, vi, vj, _di_vertex)
    dy = @dy(_di_vertex, cj)
    third = one(dy) / 3
    two_thirds = 2 * third

    if vi == ci + 1 && vj == cj
        return (εxx = third * dy, εyy = -two_thirds * dy, div = -dy)
    elseif vi == ci + 1 && vj == cj + 1
        return (εxx = -third * dy, εyy = two_thirds * dy, div = dy)
    end
    return (εxx = zero(dy), εyy = zero(dy), div = zero(dy))
end

@inline function dεxy_vertex_dVy(viτ, vjτ, vi, vj, _di_vy)
    dx = @dx(_di_vy, viτ)
    half = one(dx) / 2

    if vi == viτ && vj == vjτ
        return -half * dx
    elseif vi == viτ + 1 && vj == vjτ
        return half * dx
    end
    return zero(dx)
end

@inline function dε_center_dVy(ci, cj, vi, vj, _di_vertex, _di_vy)
    normal = dεnormal_center_dVy(ci, cj, vi, vj, _di_vertex)
    quarter = one(normal.εxx) / 4
    dεxy = quarter * (
        dεxy_vertex_dVy(ci,     cj,     vi, vj, _di_vy) +
        dεxy_vertex_dVy(ci + 1, cj,     vi, vj, _di_vy) +
        dεxy_vertex_dVy(ci,     cj + 1, vi, vj, _di_vy) +
        dεxy_vertex_dVy(ci + 1, cj + 1, vi, vj, _di_vy)
    )

    return (εxx = normal.εxx, εyy = normal.εyy, εxy = dεxy, div = normal.div)
end

@inline function dε_vertex_dVy(viτ, vjτ, vi, vj, _di_vertex, _di_vy, ni_center)
    i0, j0, ic, jc = clamped_indices(ni_center, viτ, vjτ)

    nSW = dεnormal_center_dVy(i0, j0, vi, vj, _di_vertex)
    nSE = dεnormal_center_dVy(ic, j0, vi, vj, _di_vertex)
    nNW = dεnormal_center_dVy(i0, jc, vi, vj, _di_vertex)
    nNE = dεnormal_center_dVy(ic, jc, vi, vj, _di_vertex)

    quarter = one(nSW.εxx) / 4
    dεxx = quarter * (nSW.εxx + nSE.εxx + nNW.εxx + nNE.εxx)
    dεyy = quarter * (nSW.εyy + nSE.εyy + nNW.εyy + nNE.εyy)
    dεxy = dεxy_vertex_dVy(viτ, vjτ, vi, vj, _di_vy)

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
