function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes2D_SchurComplement!(
        Dx,
        Dy,
        λmaxVx,
        λmaxVy,
        η,
        ηv,
        γ_eff,
        di.center,
        di.vertex,
        phase_ratios.vertex,
        phase_ratios.center,
        rheology,
        dt,
    )
    return nothing
end

"""
    apply_free_surface_diagonal!(Dy, λmaxVy, ρgy, di_center, dt)

Add the 2D free-surface diagonal `-dt * ∂y(ρg)` to the vertical DYREL
preconditioner and its Gershgorin row bound. Passing `dt = 0` is a no-op.
"""
function apply_free_surface_diagonal!(Dy, λmaxVy, ρgy, di_center, dt)
    ni = size(Dy)
    @parallel (@idx ni) _apply_free_surface_diagonal!(Dy, λmaxVy, ρgy, di_center, dt)
    return nothing
end

@parallel_indices (i, j) function _apply_free_surface_diagonal!(Dy, λmaxVy, ρgy, di_center, dt)
    @inbounds if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        _dy = inv(@dy(di_center, j))
        j_N = min(j + 1, size(ρgy, 2))
        c_fs = free_surface_diagonal(ρgy[i, j], ρgy[i, j_N], _dy, dt)
        D_old = Dy[i, j]
        row_sum = λmaxVy[i, j] * D_old + c_fs
        D_new = D_old + c_fs
        Dy[i, j] = D_new
        λmaxVy[i, j] = row_sum / D_new
    end
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, rheology, dt
    )


    # @inbounds begin
    phase = phase_vertex[i + 1, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[i + 1, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[i, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)

    # viscosity coefficients at surrounding points
    ηN = ηv[i + 1, j + 1]
    ηS = ηv[i + 1, j]
    ηW = η[i, j]
    # # bulk viscosity coefficients at surrounding points
    γW = γ_eff[i, j]

    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)

        # Hoist common parameters
        dx = @dx(di_center, i)
        dy = @dy(di_vertex, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        phase = phase_center[i + 1, j]
        GE = fn_ratio(get_shear_modulus, rheology, phase)
        ηE = η[i + 1, j]
        γE = γ_eff[i + 1, j]
        # effective viscoelastic viscosity
        # Equivalent to `inv(inv(η) + inv(G * dt))`, while preserving the
        # `G == Inf` limit (`ηve == η`) without producing `Inf / Inf`.
        ηN = ηN / @muladd(1 + ηN * inv(GN * dt))
        ηS = ηS / @muladd(1 + ηS * inv(GS * dt))
        ηW = ηW / @muladd(1 + ηW * inv(GW * dt))
        ηE = ηE / @muladd(1 + ηE * inv(GE * dt))

        # Precompute common terms
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        γE_dx = γE * _dx
        γW_dx = γW * _dx

        # compute Gershgorin entries
        Cxx = @muladd abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs((γE + c43 * ηE) * _dx2) +
            abs((γW + c43 * ηW) * _dx2) +
            abs((ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx)

        Cxy = @muladd abs((γE - c23 * ηE + ηN) * _dxdy) +
            abs((γE - c23 * ηE + ηS) * _dxdy) +
            abs((γW + ηN - c23 * ηW) * _dxdy) +
            abs((γW + ηS - c23 * ηW) * _dxdy)

        # this is the preconditioner diagonal entry
        Dx_ij = Dx[i, j] = @muladd (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
        # maximum eigenvalue estimate
        λmaxVx[i, j] = inv(Dx_ij) * (Cxx + Cxy)
    end

    # viscosity coefficients at surrounding points
    GS = GW # reuse cached value
    phase = phase_vertex[i, j + 1]
    GW = fn_ratio(get_shear_modulus, rheology, phase)
    GE = GN # reuse cached value

    # viscosity coefficients at surrounding points
    ηS = η[i, j]
    ηW = ηv[i, j + 1]
    ηE = ηv[i + 1, j + 1]
    # # bulk viscosity coefficients at surrounding points
    γS = γW # reuse cached value

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        # Hoist common parameters
        dx = @dx(di_vertex, i)
        dy = @dy(di_center, j)
        _dx = inv(dx)
        _dy = inv(dy)
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dxdy = _dx * _dy
        c43 = 4 / 3
        c23 = 2 / 3

        phase = phase_center[i, j + 1]
        GN = fn_ratio(get_shear_modulus, rheology, phase)

        ηN = η[i, j + 1]
        γN = γ_eff[i, j + 1]
        # effective viscoelastic viscosity
        ηN = ηN / @muladd(1 + ηN * inv(GN * dt))
        ηS = ηS / @muladd(1 + ηS * inv(GS * dt))
        ηW = ηW / @muladd(1 + ηW * inv(GW * dt))
        ηE = ηE / @muladd(1 + ηE * inv(GE * dt))

        # Precompute common terms
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        γN_dy = γN * _dy
        γS_dy = γS * _dy

        # compute Gershgorin entries
        Cyy = @muladd abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs((γN + c43 * ηN) * _dy2) +
            abs((γS + c43 * ηS) * _dy2) +
            abs((γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx)

        Cyx = @muladd abs((γN + ηE - c23 * ηN) * _dxdy) +
            abs((γN - c23 * ηN + ηW) * _dxdy) +
            abs((γS + ηE - c23 * ηS) * _dxdy) +
            abs((γS - c23 * ηS + ηW) * _dxdy)

        # this is the preconditioner diagonal entry
        Dy_ij = Dy[i, j] = @muladd (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
        # maximum eigenvalue estimate
        λmaxVy[i, j] = inv(Dy_ij) * (Cyx + Cyy)
    end
    # end

    return nothing
end

function Gershgorin_Stokes3D_SchurComplement!(Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
    ni = size(η)
    @parallel (@idx ni) _Gershgorin_Stokes3D_SchurComplement!(
        Dx,
        Dy,
        Dz,
        λmaxVx,
        λmaxVy,
        λmaxVz,
        η,
        γ_eff,
        di.center,
        di.vertex,
        phase_ratios.center,
        phase_ratios.yz,
        phase_ratios.xz,
        phase_ratios.xy,
        rheology,
        dt,
    )
    return nothing
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{2}, Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{2}, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

@inline function Gershgorin_Stokes_SchurComplement!(
        ::Val{3}, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt
    )
    return Gershgorin_Stokes3D_SchurComplement!(Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, ηv, γ_eff, phase_ratios, rheology, di, dt)
end

Base.@propagate_inbounds @inline function _ηve(ηij, rheology, phase, dt)
    Gij = fn_ratio(get_shear_modulus, rheology, phase)
    return ηij / @muladd(1 + ηij * inv(Gij * dt))
end

Base.@propagate_inbounds @inline _ηve_center(η, phase_center, rheology, dt, i, j, k) =
    _ηve(η[i, j, k], rheology, phase_center[i, j, k], dt)

Base.@propagate_inbounds @inline function _ηve_yz(η, phase_yz, rheology, dt, ni, i, j, k)
    Ic = clamped_indices(ni, i, j, k)
    return _ηve(harm_clamped_yz(η, Ic...), rheology, phase_yz[i, j, k], dt)
end

Base.@propagate_inbounds @inline function _ηve_xz(η, phase_xz, rheology, dt, ni, i, j, k)
    Ic = clamped_indices(ni, i, j, k)
    return _ηve(harm_clamped_xz(η, Ic...), rheology, phase_xz[i, j, k], dt)
end

Base.@propagate_inbounds @inline function _ηve_xy(η, phase_xy, rheology, dt, ni, i, j, k)
    Ic = clamped_indices(ni, i, j, k)
    return _ηve(harm_clamped_xy(η, Ic...), rheology, phase_xy[i, j, k], dt)
end

@parallel_indices (i, j, k) function _Gershgorin_Stokes3D_SchurComplement!(
        Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, η, γ_eff, di_center, di_vertex,
        phase_center, phase_yz, phase_xz, phase_xy, rheology, dt
    )

    ni = size(η)
    c13 = 1 / 3
    c43 = 4 / 3

    # DYREL D/λ arrays store active velocity updates; boundary values are enforced by flow_bcs! after the shifted update.
    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2) && k ≤ size(Dx, 3)
        _dx = inv(@dx(di_center, i))
        _dy = inv(@dy(di_vertex, j))
        _dz = inv(@dz(di_vertex, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdy = _dx * _dy
        _dxdz = _dx * _dz

        ηW = _ηve_center(η, phase_center, rheology, dt, i, j, k)
        ηE = _ηve_center(η, phase_center, rheology, dt, i + 1, j, k)
        ηS = _ηve_xy(η, phase_xy, rheology, dt, ni, i + 1, j, k)
        ηN = _ηve_xy(η, phase_xy, rheology, dt, ni, i + 1, j + 1, k)
        ηB = _ηve_xz(η, phase_xz, rheology, dt, ni, i + 1, j, k)
        ηF = _ηve_xz(η, phase_xz, rheology, dt, ni, i + 1, j, k + 1)
        γW = γ_eff[i, j, k]
        γE = γ_eff[i + 1, j, k]

        Dx_ijk = Dx[i, j, k] = @muladd(
            (ηN + ηS) * _dy2 +
            (ηB + ηF) * _dz2 +
            (γE + γW + c43 * (ηE + ηW)) * _dx2
        )

        Cx = @muladd(
            abs((γE + c43 * ηE) * _dx2) +
            abs((γW + c43 * ηW) * _dx2) +
            abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs(ηB * _dz2) +
            abs(ηF * _dz2) +
            abs((γE - (2 / 3) * ηE + ηN) * _dxdy) +
            abs((γE - (2 / 3) * ηE + ηS) * _dxdy) +
            abs((γW + ηN - (2 / 3) * ηW) * _dxdy) +
            abs((γW + ηS - (2 / 3) * ηW) * _dxdy) +
            abs((γE - (2 / 3) * ηE + ηB) * _dxdz) +
            abs((γW + ηB - (2 / 3) * ηW) * _dxdz) +
            abs((γE - (2 / 3) * ηE + ηF) * _dxdz) +
            abs((γW + ηF - (2 / 3) * ηW) * _dxdz) +
            abs(Dx_ijk)
        )

        λmaxVx[i, j, k] = Cx / Dx_ijk
    end

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2) && k ≤ size(Dy, 3)
        _dx = inv(@dx(di_vertex, i))
        _dy = inv(@dy(di_center, j))
        _dz = inv(@dz(di_vertex, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdy = _dx * _dy
        _dydz = _dy * _dz

        ηW = _ηve_xy(η, phase_xy, rheology, dt, ni, i, j + 1, k)
        ηE = _ηve_xy(η, phase_xy, rheology, dt, ni, i + 1, j + 1, k)
        ηS = _ηve_center(η, phase_center, rheology, dt, i, j, k)
        ηN = _ηve_center(η, phase_center, rheology, dt, i, j + 1, k)
        ηB = _ηve_yz(η, phase_yz, rheology, dt, ni, i, j + 1, k)
        ηF = _ηve_yz(η, phase_yz, rheology, dt, ni, i, j + 1, k + 1)
        γS = γ_eff[i, j, k]
        γN = γ_eff[i, j + 1, k]

        Dy_ijk = Dy[i, j, k] = @muladd(
            (ηE + ηW) * _dx2 +
            (ηB + ηF) * _dz2 +
            (γN + γS + c43 * (ηN + ηS)) * _dy2
        )

        Cy = @muladd(
            abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs((γN + c43 * ηN) * _dy2) +
            abs((γS + c43 * ηS) * _dy2) +
            abs(ηB * _dz2) +
            abs(ηF * _dz2) +
            abs((γN + ηE - (2 / 3) * ηN) * _dxdy) +
            abs((γS + ηE - (2 / 3) * ηS) * _dxdy) +
            abs((γN - (2 / 3) * ηN + ηW) * _dxdy) +
            abs((γS - (2 / 3) * ηS + ηW) * _dxdy) +
            abs((γN - (2 / 3) * ηN + ηB) * _dydz) +
            abs((γS - (2 / 3) * ηS + ηB) * _dydz) +
            abs((γN - (2 / 3) * ηN + ηF) * _dydz) +
            abs((γS - (2 / 3) * ηS + ηF) * _dydz) +
            abs(Dy_ijk)
        )

        λmaxVy[i, j, k] = Cy / Dy_ijk
    end

    if i ≤ size(Dz, 1) && j ≤ size(Dz, 2) && k ≤ size(Dz, 3)
        _dx = inv(@dx(di_vertex, i))
        _dy = inv(@dy(di_vertex, j))
        _dz = inv(@dz(di_center, k))
        _dx2 = _dx * _dx
        _dy2 = _dy * _dy
        _dz2 = _dz * _dz
        _dxdz = _dx * _dz
        _dydz = _dy * _dz

        ηW = _ηve_xz(η, phase_xz, rheology, dt, ni, i, j, k + 1)
        ηE = _ηve_xz(η, phase_xz, rheology, dt, ni, i + 1, j, k + 1)
        ηS = _ηve_yz(η, phase_yz, rheology, dt, ni, i, j, k + 1)
        ηN = _ηve_yz(η, phase_yz, rheology, dt, ni, i, j + 1, k + 1)
        ηB = _ηve_center(η, phase_center, rheology, dt, i, j, k)
        ηF = _ηve_center(η, phase_center, rheology, dt, i, j, k + 1)
        γB = γ_eff[i, j, k]
        γF = γ_eff[i, j, k + 1]

        Dz_ijk = Dz[i, j, k] = @muladd(
            (ηE + ηW) * _dx2 +
            (ηN + ηS) * _dy2 +
            (γB + γF + c43 * (ηB + ηF)) * _dz2
        )

        Cz = @muladd(
            abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs((γB + c43 * ηB) * _dz2) +
            abs((γF + c43 * ηF) * _dz2) +
            abs((γB - (2 / 3) * ηB + ηE) * _dxdz) +
            abs((γB - (2 / 3) * ηB + ηW) * _dxdz) +
            abs((γF - (2 / 3) * ηF + ηE) * _dxdz) +
            abs((γF - (2 / 3) * ηF + ηW) * _dxdz) +
            abs((γB - (2 / 3) * ηB + ηN) * _dydz) +
            abs((γB - (2 / 3) * ηB + ηS) * _dydz) +
            abs((γF - (2 / 3) * ηF + ηN) * _dydz) +
            abs((γF - (2 / 3) * ηF + ηS) * _dydz) +
            abs(Dz_ijk)
        )

        λmaxVz[i, j, k] = Cz / Dz_ijk
    end

    return nothing
end

"""
    update_α_β!(βV, αV, dτV, cV)

Generic N-dimensional version (2D and 3D) of the acceleration parameters update.

Updates the damping parameters `βV` and `αV` for each velocity component based on
the pseudo-time step `dτV` and the preconditioner diagonal `cV`.

# Arguments
- `βV::NTuple{N, AbstractArray{T, N}}`: Tuple of damping parameters β for each velocity component
- `αV::NTuple{N, AbstractArray{T, N}}`: Tuple of acceleration parameters α for each velocity component
- `dτV::NTuple{N, AbstractArray{T, N}}`: Tuple of pseudo-time steps for each velocity component
- `cV::NTuple{N, AbstractArray{T, N}}`: Tuple of preconditioner diagonal entries for each velocity component
"""
function update_α_β!(
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}
    ni = size(βV[1]) .+ ntuple(i -> i == 1 ? 1 : 0, Val(N))
    @parallel (@idx ni) _update_α_β!(βV, αV, dτV, cV)
    return nothing
end

@parallel_indices (I...) function _update_α_β!(
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        dτV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}}
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(βV[i]))
            dτV_ij = dτV[i][I...]
            cV_ij = cV[i][I...]
            cdt = cV_ij * dτV_ij
            inv_den = inv(2 + cdt)
            βV[i][I...] = @muladd 2 * dτV_ij * inv_den
            αV[i][I...] = @muladd (2 - cdt) * inv_den
        end
    end
    return nothing
end

"""
    update_dτV_α_β!(dτV, βV, αV, cV, λmaxV, CFL_v)

Generic N-dimensional version (2D and 3D) for updating pseudo-time step and acceleration parameters.

Computes the pseudo-time step `dτV` from the maximum eigenvalue estimate `λmaxV` and CFL number,
then updates the damping parameters `βV` and acceleration parameters `αV` accordingly.

# Arguments
- `dτV::NTuple{N, AbstractArray{T, N}}`: Tuple of pseudo-time steps for each velocity component
- `βV::NTuple{N, AbstractArray{T, N}}`: Tuple of damping parameters β for each velocity component
- `αV::NTuple{N, AbstractArray{T, N}}`: Tuple of acceleration parameters α for each velocity component
- `cV::NTuple{N, AbstractArray{T, N}}`: Tuple of preconditioner diagonal entries for each velocity component
- `λmaxV::NTuple{N, AbstractArray{T, N}}`: Tuple of maximum eigenvalue estimates for each velocity component
- `CFL_v::Real`: CFL number for velocity
"""
function update_dτV_α_β!(
        dτV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}},
        λmaxV::NTuple{N, AbstractArray{T, N}},
        CFL_v::Real
    ) where {N, T}
    ni = size(βV[1]) .+ ntuple(i -> i == 1 ? 1 : 0, Val(N))
    @parallel (@idx ni) _update_dτV_α_β!(dτV, βV, αV, cV, λmaxV, CFL_v)
    return nothing
end

@parallel_indices (I...) function _update_dτV_α_β!(
        dτV::NTuple{N, AbstractArray{T, N}},
        βV::NTuple{N, AbstractArray{T, N}},
        αV::NTuple{N, AbstractArray{T, N}},
        cV::NTuple{N, AbstractArray{T, N}},
        λmaxV::NTuple{N, AbstractArray{T, N}},
        CFL_v::Real
    ) where {N, T}
    ntuple(Val(N)) do i
        @inline
        if all(I .≤ size(βV[i]))
            dτV_ij = dτV[i][I...] = 2 / √(λmaxV[i][I...]) * CFL_v
            cV_ij = cV[i][I...]
            cdt = cV_ij * dτV_ij
            inv_den = inv(2 + cdt)
            βV[i][I...] = @muladd 2 * dτV_ij * inv_den
            αV[i][I...] = @muladd (2 - cdt) * inv_den
        end
    end
    return nothing
end

function update_α_β!(dyrel::JustRelax.DYREL)
    return update_α_β!(Val(ndims(dyrel.γ_eff)), dyrel)
end

function update_α_β!(::Val{2}, dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.cVx, dyrel.cVy)
    )
end

function update_α_β!(::Val{3}, dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        (dyrel.αVx, dyrel.αVy, dyrel.αVz),
        (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        (dyrel.cVx, dyrel.cVy, dyrel.cVz)
    )
end

function update_dτV_α_β!(dyrel::JustRelax.DYREL)
    return update_dτV_α_β!(dyrel, dyrel.CFL)
end

function update_dτV_α_β!(dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(Val(ndims(dyrel.γ_eff)), dyrel, CFL_v)
end

function update_dτV_α_β!(::Val{2}, dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.cVx, dyrel.cVy),
        (dyrel.λmaxVx, dyrel.λmaxVy),
        CFL_v
    )
end

function update_dτV_α_β!(::Val{3}, dyrel::JustRelax.DYREL, CFL_v)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        (dyrel.αVx, dyrel.αVy, dyrel.αVz),
        (dyrel.cVx, dyrel.cVy, dyrel.cVz),
        (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
        CFL_v
    )
end

# 2D wrapper for update_α_β! with individual arguments
function update_α_β!(βVx, βVy, αVx, αVy, dτVx, dτVy, cVx, cVy)
    return update_α_β!(
        (βVx, βVy),
        (αVx, αVy),
        (dτVx, dτVy),
        (cVx, cVy)
    )
end

# 3D wrapper for update_α_β! with individual arguments
function update_α_β!(βVx, βVy, βVz, αVx, αVy, αVz, dτVx, dτVy, dτVz, cVx, cVy, cVz)
    return update_α_β!(
        (βVx, βVy, βVz),
        (αVx, αVy, αVz),
        (dτVx, dτVy, dτVz),
        (cVx, cVy, cVz)
    )
end

# 2D wrapper for update_dτV_α_β! with individual arguments
function update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)
    return update_dτV_α_β!(
        (dτVx, dτVy),
        (βVx, βVy),
        (αVx, αVy),
        (cVx, cVy),
        (λmaxVx, λmaxVy),
        CFL_v
    )
end

# 3D wrapper for update_dτV_α_β! with individual arguments
function update_dτV_α_β!(dτVx, dτVy, dτVz, βVx, βVy, βVz, αVx, αVy, αVz, cVx, cVy, cVz, λmaxVx, λmaxVy, λmaxVz, CFL_v)
    return update_dτV_α_β!(
        (dτVx, dτVy, dτVz),
        (βVx, βVy, βVz),
        (αVx, αVy, αVz),
        (cVx, cVy, cVz),
        (λmaxVx, λmaxVy, λmaxVz),
        CFL_v
    )
end
