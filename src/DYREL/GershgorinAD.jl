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

# Local velocity stencils for one Rx[i,j] residual Gershgorin estimate.
#
# The x=... and y=... labels below are velocity-array indices, not
# physical coordinates. For example, y=j+1 means the row Vx[:, j+1]
# or Vy[:, j+1], depending on the stencil.
#
# Vx stencil:
#
#   k=7 ---- k=8 ---- k=9      y = j+2
#    |        |        |
#   k=4 ---- k=5 ---- k=6      y = j+1     Rx[i,j] is at k=5
#    |        |        |
#   k=1 ---- k=2 ---- k=3      y = j
#   x=i     x=i+1    x=i+2
#
# Vy stencil:
#
#   k=9 ---- k=10 --- k=11 --- k=12     y = j+2
#    |        |        |        |
#   k=5 ---- k=6  --- k=7  --- k=8      y = j+1     Rx[i,j] lies between k=6 and k=7
#    |        |        |        |
#   k=1 ---- k=2  --- k=3  --- k=4      y = j
#   x=i     x=i+1    x=i+2    x=i+3

@parallel_indices (I...) function assemble_Rx_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dx, 1) && j ≤ size(dyrel.Dx, 2)
        ni_center = size(dyrel.γ_eff)

        Cxx = zero(eltype(dyrel.Dx))
        for k in 1:9
            gershgorin_entry = local_Rx_Vx_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vx, ni_center)
            Cxx += gershgorin_entry
            if k == 5
                dyrel.Dx[i, j] = gershgorin_entry
            end
        end

        Cxy = zero(eltype(dyrel.Dx))
        for k in 1:12
            gershgorin_entry = local_Rx_Vy_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vy, ni_center)
            Cxy += gershgorin_entry
        end

        dyrel.λmaxVx[i, j] = inv(dyrel.Dx[i, j]) * (Cxx + Cxy)
    end

    return nothing
end

# Local velocity stencils for one Ry[i,j] residual Gershgorin estimate.
#
# The x=... and y=... labels below are velocity-array indices, not
# physical coordinates. For example, y=j+1 means the row Vx[:, j+1]
# or Vy[:, j+1], depending on the stencil.
#
# Vx stencil:
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
# Vy stencil:
#
#   k=7 ---- k=8 ---- k=9      y = j+2
#    |        |        |
#   k=4 ---- k=5 ---- k=6      y = j+1     Ry[i,j] is at k=5
#    |        |        |
#   k=1 ---- k=2 ---- k=3      y = j
#   x=i     x=i+1    x=i+2
#
@parallel_indices (I...) function assemble_Ry_gershgorin!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I

    if i ≤ size(dyrel.Dy, 1) && j ≤ size(dyrel.Dy, 2)
        ni_center = size(dyrel.γ_eff)

        Cyx = zero(eltype(dyrel.Dy))
        for k in 1:12
            gershgorin_entry = local_Ry_Vx_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vx, ni_center)
            Cyx += gershgorin_entry
        end

        Cyy = zero(eltype(dyrel.Dy))
        for k in 1:9
            gershgorin_entry = local_Ry_Vy_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vy, ni_center)
            Cyy += gershgorin_entry
            if k == 5
                dyrel.Dy[i, j] = gershgorin_entry
            end
        end

        dyrel.λmaxVy[i, j] = inv(dyrel.Dy[i, j]) * (Cyx + Cyy)
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

@inline function local_Rx_Vx_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vx, ni_center)
    _dx = @dx(_di_center, i)
    _dy = @dy(_di_vertex, j)
    vi, vj = local_Rx_Vx_index(i, j, k)

    εW = dε_center_dVx(i,     j, vi, vj, _di_vertex, _di_vx)
    εE = dε_center_dVx(i + 1, j, vi, vj, _di_vertex, _di_vx)
    εS = dε_vertex_dVx(i + 1, j,     vi, vj, _di_vertex, _di_vx, ni_center)
    εN = dε_vertex_dVx(i + 1, j + 1, vi, vj, _di_vertex, _di_vx, ni_center)

    dτxxW = dτ_dV(dyrel.∂τc_∂ε, 1, i,     j, εW.εxx, εW.εyy, εW.εxy)
    dτxxE = dτ_dV(dyrel.∂τc_∂ε, 1, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
    dτxyS = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j,     εS.εxx, εS.εyy, εS.εxy)
    dτxyN = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εN.εxx, εN.εyy, εN.εxy)

    dΔPψW = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i,     j, εW.εxx, εW.εyy, εW.εxy)
    dΔPψE = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
    dPnumW = dyrel.γ_eff[i,     j] * εW.div
    dPnumE = dyrel.γ_eff[i + 1, j] * εE.div

    τxx_term = _dx * (dτxxE - dτxxW)
    τxy_term = _dy * (dτxyN - dτxyS)
    Pnum_term = -_dx * (dPnumE - dPnumW)
    ΔPψ_term = -_dx * (dΔPψE - dΔPψW)
    return abs(τxx_term) + abs(τxy_term) + abs(Pnum_term) + abs(ΔPψ_term)
end

@inline function local_Rx_Vy_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vy, ni_center)
    _dx = @dx(_di_center, i)
    _dy = @dy(_di_vertex, j)
    vi, vj = local_Rx_Vy_index(i, j, k)

    εW = dε_center_dVy(i,     j, vi, vj, _di_vertex, _di_vy)
    εE = dε_center_dVy(i + 1, j, vi, vj, _di_vertex, _di_vy)
    εS = dε_vertex_dVy(i + 1, j,     vi, vj, _di_vertex, _di_vy, ni_center)
    εN = dε_vertex_dVy(i + 1, j + 1, vi, vj, _di_vertex, _di_vy, ni_center)

    dτxxW = dτ_dV(dyrel.∂τc_∂ε, 1, i,     j, εW.εxx, εW.εyy, εW.εxy)
    dτxxE = dτ_dV(dyrel.∂τc_∂ε, 1, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
    dτxyS = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j,     εS.εxx, εS.εyy, εS.εxy)
    dτxyN = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εN.εxx, εN.εyy, εN.εxy)

    dΔPψW = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i,     j, εW.εxx, εW.εyy, εW.εxy)
    dΔPψE = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i + 1, j, εE.εxx, εE.εyy, εE.εxy)
    dPnumW = dyrel.γ_eff[i,     j] * εW.div
    dPnumE = dyrel.γ_eff[i + 1, j] * εE.div

    τxx_term = _dx * (dτxxE - dτxxW)
    τxy_term = _dy * (dτxyN - dτxyS)
    Pnum_term = -_dx * (dPnumE - dPnumW)
    ΔPψ_term = -_dx * (dΔPψE - dΔPψW)
    return abs(Pnum_term + ΔPψ_term) - abs(τxx_term) + abs(τxy_term)
end

@inline function local_Ry_Vx_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vx, ni_center)
    _dy = @dy(_di_center, j)
    _dx = @dx(_di_vertex, i)
    vi, vj = local_Ry_Vx_index(i, j, k)

    εS = dε_center_dVx(i, j,     vi, vj, _di_vertex, _di_vx)
    εN = dε_center_dVx(i, j + 1, vi, vj, _di_vertex, _di_vx)
    εW = dε_vertex_dVx(i,     j + 1, vi, vj, _di_vertex, _di_vx, ni_center)
    εE = dε_vertex_dVx(i + 1, j + 1, vi, vj, _di_vertex, _di_vx, ni_center)

    dτyyS = dτ_dV(dyrel.∂τc_∂ε, 2, i, j,     εS.εxx, εS.εyy, εS.εxy)
    dτyyN = dτ_dV(dyrel.∂τc_∂ε, 2, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
    dτxyW = dτ_dV(dyrel.∂τv_∂ε, 3, i,     j + 1, εW.εxx, εW.εyy, εW.εxy)
    dτxyE = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εE.εxx, εE.εyy, εE.εxy)

    dΔPψS = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j,     εS.εxx, εS.εyy, εS.εxy)
    dΔPψN = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
    dPnumS = dyrel.γ_eff[i, j]     * εS.div
    dPnumN = dyrel.γ_eff[i, j + 1] * εN.div

    τyy_term = _dy * (dτyyN - dτyyS)
    τxy_term = _dx * (dτxyE - dτxyW)
    Pnum_term = -_dy * (dPnumN - dPnumS)
    ΔPψ_term = -_dy * (dΔPψN - dΔPψS)
    return abs(Pnum_term + ΔPψ_term) - abs(τyy_term) + abs(τxy_term)
end

@inline function local_Ry_Vy_gershgorin_entry(dyrel, i, j, k, _di_center, _di_vertex, _di_vy, ni_center)
    _dy = @dy(_di_center, j)
    _dx = @dx(_di_vertex, i)
    vi, vj = local_Ry_Vy_index(i, j, k)

    εS = dε_center_dVy(i, j,     vi, vj, _di_vertex, _di_vy)
    εN = dε_center_dVy(i, j + 1, vi, vj, _di_vertex, _di_vy)
    εW = dε_vertex_dVy(i,     j + 1, vi, vj, _di_vertex, _di_vy, ni_center)
    εE = dε_vertex_dVy(i + 1, j + 1, vi, vj, _di_vertex, _di_vy, ni_center)

    dτyyS = dτ_dV(dyrel.∂τc_∂ε, 2, i, j,     εS.εxx, εS.εyy, εS.εxy)
    dτyyN = dτ_dV(dyrel.∂τc_∂ε, 2, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
    dτxyW = dτ_dV(dyrel.∂τv_∂ε, 3, i,     j + 1, εW.εxx, εW.εyy, εW.εxy)
    dτxyE = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, εE.εxx, εE.εyy, εE.εxy)

    dΔPψS = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j,     εS.εxx, εS.εyy, εS.εxy)
    dΔPψN = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i, j + 1, εN.εxx, εN.εyy, εN.εxy)
    dPnumS = dyrel.γ_eff[i, j]     * εS.div
    dPnumN = dyrel.γ_eff[i, j + 1] * εN.div

    τyy_term = _dy * (dτyyN - dτyyS)
    τxy_term = _dx * (dτxyE - dτxyW)
    Pnum_term = -_dy * (dPnumN - dPnumS)
    ΔPψ_term = -_dy * (dΔPψN - dΔPψS)
    return abs(τyy_term) + abs(τxy_term) + abs(Pnum_term) + abs(ΔPψ_term)
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
