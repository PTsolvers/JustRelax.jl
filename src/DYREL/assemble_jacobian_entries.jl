function assemble_jacobian(
    dyrel,
    _di_center,
    _di_vertex,
    _di_vx,
    _di_vy,
    )

    ni = size(dyrel.γ_eff)
    @parallel (@idx ni)  assemble_Rx!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)
    return nothing
end

@parallel_indices (I...) function assemble_Rx!(dyrel, _di_center, _di_vertex, _di_vx, _di_vy)

    i, j = I
    # get local grid spacing
    _dx = @dx(_di_center, i)
    _dy = @dy(_di_vertex, j)

    if i ≤ size(dyrel.∂Rx_∂Vx[1], 1) && j ≤ size(dyrel.∂Rx_∂Vx[1], 2)

        # Vx5 = Vx[i+1, j+1]. # central velocity point
        # grid spacing in the cell rigth of the current Rx point
        dxW = @dx(_di_vertex, i)
        dxE = @dx(_di_vertex, i + 1)

        dεxxW_dVx =  (2 / 3) * dxW
        dεyyW_dVx = -(1 / 3) * dxW

        dεxxE_dVx = -(2 / 3) * dxE
        dεyyE_dVx =  (1 / 3) * dxE

        dyS = @dy(_di_vx, j)
        dyN = @dy(_di_vx, j + 1)

        # ∂εxy_west/∂Vx & ∂εxy_east/∂Vx.   !! Here we consider εxy which is interpolated to the center from the vertexes around !!
        dεxyW_cen_dVx = 0.125 * (dyS - dyN)
        dεxyE_cen_dVx = 0.125 * (dyS - dyN)

        dεxxN_dVx = 0.25 * (dεxxW_dVx + dεxxE_dVx)
        dεyyN_dVx = 0.25 * (dεyyW_dVx + dεyyE_dVx)
        dεxxS_dVx = dεxxN_dVx
        dεyyS_dVx = dεyyN_dVx

        # ∂εxy_north/∂Vx & ∂εxy_south/∂Vx
        dεxyN_dVx = -0.5 * dyN
        dεxyS_dVx =  0.5 * dyS

        dτxxW_dVx = dτ_dV(dyrel.∂τc_∂ε, 1, i,     j,     dεxxW_dVx, dεyyW_dVx, dεxyW_cen_dVx)
        dτxxE_dVx = dτ_dV(dyrel.∂τc_∂ε, 1, i + 1, j,     dεxxE_dVx, dεyyE_dVx, dεxyE_cen_dVx)
        dτxyN_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j + 1, dεxxN_dVx, dεyyN_dVx, dεxyN_dVx)
        dτxyS_dVx = dτ_dV(dyrel.∂τv_∂ε, 3, i + 1, j,     dεxxS_dVx, dεyyS_dVx, dεxyS_dVx)

        # pressure correction term
        ΔPψW_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i,     j, dεxxW_dVx, dεyyW_dVx, dεxyW_cen_dVx)
        ΔPψE_dVx = dΔPψ_dV(dyrel.∂ΔPψc_∂ε, i + 1, j, dεxxE_dVx, dεyyE_dVx, dεxyE_cen_dVx)

        # numerical pressure term: ∂P_num/∂x
        γ_effW     = dyrel.γ_eff[i, j]
        γ_effE     = dyrel.γ_eff[i+1, j]
        dPnumW_dVx =  γ_effW * dxW
        dPnumE_dVx = -γ_effE * dxE

        # assemble final gradient
        dyrel.∂Rx_∂Vx[5][i, j] = _dx * (dτxxE_dVx - dτxxW_dVx) + _dy * (dτxyN_dVx - dτxyS_dVx) - _dx * (dPnumE_dVx - dPnumW_dVx) - _dx * (ΔPψE_dVx - ΔPψW_dVx)
    end

    return nothing
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
