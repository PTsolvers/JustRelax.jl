
function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di..., phase_ratios.vertex, phase_ratios.center, rheology, dt)
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, dx, dy,
    phase_vertex, phase_center, rheology, dt
)
    # @inbounds begin
        phase = phase_vertex[i+1, j+1]
        GN = fn_ratio(get_shear_modulus, rheology, phase)
        phase = phase_vertex[i+1, j]
        GS = fn_ratio(get_shear_modulus, rheology, phase)
        phase = phase_center[i, j]
        GW = fn_ratio(get_shear_modulus, rheology, phase)

        # viscosity coefficients at surrounding points
        ηN = ηv[i+1, j+1]
        ηS = ηv[i+1, j]
        ηW = η[i, j]
        # # bulk viscosity coefficients at surrounding points
        γW = γ_eff[i, j]
        
        if all( (i,j) .≤ size(Dx) )
            phase = phase_center[i+1, j]
            GE = fn_ratio(get_shear_modulus, rheology, phase)
            ηE = η[i+1, j]
            γE = γ_eff[i+1, j]
            # effective viscoelastic viscosity 
            ηN = 1 / (1 / ηN + 1 / (GN * dt))
            ηS = 1 / (1 / ηS + 1 / (GS * dt))
            ηW = 1 / (1 / ηW + 1 / (GW * dt))
            ηE = 1 / (1 / ηE + 1 / (GE * dt))
            # compute Gershgorin entries
            Cxx = abs(ηN / dy ^ 2) + abs(ηS / dy ^ 2) + abs(γE / dx ^ 2 + (4 / 3) * ηE / dx ^ 2) + abs(γW / dx ^ 2 + (4 / 3) * ηW / dx ^ 2) + abs(-(-ηN / dy - ηS / dy) / dy + (γE / dx + γW / dx) / dx + ((4 / 3) * ηE / dx + (4 / 3) * ηW / dx) / dx)
            Cxy = abs(γE / (dx * dy) - 2 / 3 * ηE / (dx * dy) + ηN / (dx * dy)) + abs(γE / (dx * dy) - 2 / 3 * ηE / (dx * dy) + ηS / (dx * dy)) + abs(γW / (dx * dy) + ηN / (dx * dy) - 2 / 3 * ηW / (dx * dy)) + abs(γW / (dx * dy) + ηS / (dx * dy) - 2 / 3 * ηW / (dx * dy))
            # this is the preconditioner diagonal entry
            Dx_ij = Dx[i, j] = -(-ηN / dy - ηS / dy) / dy + (γE / dx + γW / dx) / dx + ((4 / 3) * ηE / dx + (4 / 3) * ηW / dx) / dx
            # maximum eigenvalue estimate
            λmaxVx[i, j] = inv(Dx_ij) * (Cxx + Cxy)
        end

        
        # viscosity coefficients at surrounding points
        GS = GW # reuse cached value
        phase = phase_vertex[i, j+1]
        GW = fn_ratio(get_shear_modulus, rheology, phase)
        GE = GN # reuse cached value

        # viscosity coefficients at surrounding points
        ηS = η[i, j]
        ηW = ηv[i, j+1]
        ηE = ηv[i+1, j+1] 
        # # bulk viscosity coefficients at surrounding points
        γS = γW # reuse cached value
        
        if all( (i,j) .≤ size(Dy) )
            phase = phase_center[i, j+1]
            GN = fn_ratio(get_shear_modulus, rheology, phase)
      
            ηN = η[i, j+1]
            γN = γ_eff[i, j+1]
            # effective viscoelastic viscosity 
            ηN = 1 / (1 / ηN + 1 / (GN * dt))
            ηS = 1 / (1 / ηS + 1 / (GS * dt))
            ηW = 1 / (1 / ηW + 1 / (GW * dt))
            ηE = 1 / (1 / ηE + 1 / (GE * dt))
            # compute Gershgorin entries
            Cyy = abs(ηE / dx ^ 2) + abs(ηW / dx ^ 2) + abs(γN / dy ^ 2 + (4 / 3) * ηN / dy ^ 2) + abs(γS / dy ^ 2 + (4 / 3) * ηS / dy ^ 2) + abs((γN / dy + γS / dy) / dy + ((4 / 3) * ηN / dy + (4 / 3) * ηS / dy) / dy - (-ηE / dx - ηW / dx) / dx)
            Cyx = abs(γN / (dx * dy) + ηE / (dx * dy) - 2 / 3 * ηN / (dx * dy)) + abs(γN / (dx * dy) - 2 / 3 * ηN / (dx * dy) + ηW / (dx * dy)) + abs(γS / (dx * dy) + ηE / (dx * dy) - 2 / 3 * ηS / (dx * dy)) + abs(γS / (dx * dy) - 2 / 3 * ηS / (dx * dy) + ηW / (dx * dy))
            # this is the preconditioner diagonal entry
            Dy_ij = Dy[i, j] = (γN / dy + γS / dy) / dy + ((4 / 3) * ηN / dy + (4 / 3) * ηS / dy) / dy - (-ηE / dx - ηW / dx) / dx
            # maximum eigenvalue estimate
            λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
        end    
    # end

    return nothing
end

function update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    ni = size(βVx) .+ (1, 0)
    @parallel (@idx ni) _update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    return nothing
end

@parallel_indices (I...) function _update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)

    if all(I .≤ size(βVx))
        dτVx_ij = dτVx[I...]
        cVx_ij = cVx[I...]

        βVx[I...] = @muladd  2 * dτVx_ij / (2 + cVx_ij * dτVx_ij)
        αVx[I...] = @muladd (2 - cVx_ij * dτVx_ij) / (2 + cVx_ij * dτVx_ij)
    end
    if all(I .≤ size(βVy))
        dτVy_ij = dτVy[I...]
        cVy_ij  = cVy[I...]
        βVy[I...] = @muladd  2 * dτVy_ij / (2 + cVy_ij *dτVy_ij)
        αVy[I...] = @muladd (2 - cVy_ij * dτVy_ij) / (2 + cVy_ij *dτVy_ij)
    end

    return nothing
end

    
function update_dτV_α_β!(dyrel::DYREL)
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.CFL)
end

function update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)
    ni = size(βVx) .+ (1, 0)
    @parallel (@idx ni) _update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)
    return nothing
end

@parallel_indices (I...) function _update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)

    if all(I .≤ size(βVx))

        dτVx_ij = dτVx[I...] =  2 / √(λmaxVx[I...]) * CFL_v

        dτVx_ij = dτVx[I...]
        cVx_ij  = cVx[I...]

        βVx[I...] = @muladd  2 * dτVx_ij / (2 + cVx_ij * dτVx_ij)
        αVx[I...] = @muladd (2 - cVx_ij * dτVx_ij) / (2 + cVx_ij * dτVx_ij)
    end

    if all(I .≤ size(βVy))
        dτVy_ij = dτVy[I...] =  2 / √(λmaxVy[I...]) * CFL_v

        dτVy_ij = dτVy[I...]
        cVy_ij  = cVy[I...]
        βVy[I...] = @muladd  2 * dτVy_ij / (2 + cVy_ij *dτVy_ij)
        αVy[I...] = @muladd (2 - cVy_ij * dτVy_ij) / (2 + cVy_ij *dτVy_ij)
    end

    return nothing
end