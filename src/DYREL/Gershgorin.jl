# Diagonal contribution of the free-surface stabilization buoyancy term (Kaus et al.,
# 2010) to the Vy operator: ∂(ρg)/∂y·θ·dt, with θ = 1 to match the residual kernels
# (`compute_PH_residual_V!`/`compute_DR_residual_update_V!`), which add Vy·∂(ρg)/∂y·θ·dt
# to Ry. Passing a `RockRatio` selects the ϕ-masked ρg sampling those masked kernels use, so
# the preconditioner describes the same stabilized operator. Returns a Bool `false` (numeric
# zero) when no buoyancy field is supplied, so construction-time estimates and the FSSA-off
# path are byte-identical to the plain viscous diagonal.
@inline fssa_diagonal_y(::Nothing, i, j, _dy, dt, ϕ = nothing) = false
Base.@propagate_inbounds @inline function fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ = nothing)
    # the launch spans every center row, so the north neighbour of the top row does not exist
    j == lastindex(ρgy, 2) && return zero(eltype(ρgy))
    return _d_ya_ρg(ρgy, ϕ, _dy, i, j) * dt
end

Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ::Nothing, _dy, i, j) = _d_ya(ρgy, _dy, i, j)
Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ϕ::JustRelax.RockRatio, _dy, i, j) = _d_ya(ρgy, ϕ.center, _dy, i, j)

# Store a preconditioner diagonal and its eigenvalue bound. A valid diagonal is strictly
# positive; it can still come out zero or NaN at a mask-boundary cell that `isvalid_*` accepts
# but whose phase ratio or ϕ configuration is degenerate, and `inv` would then poison both the
# bound and the velocity update. Such a cell is treated like an invalid one.
Base.@propagate_inbounds @inline function set_preconditioner!(D, λmax, D_ij, C, i, j)
    if D_ij > 0
        D[i, j] = D_ij
        λmax[i, j] = inv(D_ij) * C
    else
        D[i, j] = one(eltype(D))
        λmax[i, j] = one(eltype(λmax))
    end
    return nothing
end

function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, rheology, di, dt, ρgy = nothing)
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
        ρgy,
    )
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, rheology, dt, ρgy
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
        ηN = 1 / (1 / ηN + 1 / (GN * dt))
        ηS = 1 / (1 / ηS + 1 / (GS * dt))
        ηW = 1 / (1 / ηW + 1 / (GW * dt))
        ηE = 1 / (1 / ηE + 1 / (GE * dt))

        # Precompute common terms
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        γE_dx = γE * _dx
        γW_dx = γW * _dx

        # compute Gershgorin entries
        Cxx = abs(ηN * _dy2) +
            abs(ηS * _dy2) +
            abs((γE + c43 * ηE) * _dx2) +
            abs((γW + c43 * ηW) * _dx2) +
            abs((ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx)

        Cxy = abs((γE - c23 * ηE + ηN) * _dxdy) +
            abs((γE - c23 * ηE + ηS) * _dxdy) +
            abs((γW + ηN - c23 * ηW) * _dxdy) +
            abs((γW + ηS - c23 * ηW) * _dxdy)

        # this is the preconditioner diagonal entry
        Dx_ij = Dx[i, j] = (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
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
        ηN = 1 / (1 / ηN + 1 / (GN * dt))
        ηS = 1 / (1 / ηS + 1 / (GS * dt))
        ηW = 1 / (1 / ηW + 1 / (GW * dt))
        ηE = 1 / (1 / ηE + 1 / (GE * dt))

        # Precompute common terms
        ηE_dx = ηE * _dx
        ηW_dx = ηW * _dx
        ηN_dy = ηN * _dy
        ηS_dy = ηS * _dy
        γN_dy = γN * _dy
        γS_dy = γS * _dy

        # Viscous+penalty diagonal, augmented by the free-surface stabilization term.
        # The Ry residual carries an implicit buoyancy term Vy·∂(ρg)/∂y·θ·dt (Kaus et
        # al., 2010), whose diagonal contribution must enter the preconditioner/eigenvalue
        # bound so they describe the stabilized operator the iteration actually solves.
        # Use the over-bound Dy_visc + |FSSA|: it never cancels to zero (which would trip
        # the degenerate-diagonal guard below into a destabilizing dτ) and is a valid
        # Gershgorin bound — it can only make dτ smaller/safer.
        Dy_visc = (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
        Dy_mag = Dy_visc + abs(fssa_diagonal_y(ρgy, i, j, _dy, dt))

        # compute Gershgorin entries
        Cyy = abs(ηE * _dx2) +
            abs(ηW * _dx2) +
            abs((γN + c43 * ηN) * _dy2) +
            abs((γS + c43 * ηS) * _dy2) +
            Dy_mag

        Cyx = abs((γN + ηE - c23 * ηN) * _dxdy) +
            abs((γN - c23 * ηN + ηW) * _dxdy) +
            abs((γS + ηE - c23 * ηS) * _dxdy) +
            abs((γS - c23 * ηS + ηW) * _dxdy)

        set_preconditioner!(Dy, λmaxVy, Dy_mag, Cyx + Cyy, i, j)
    end
    # end

    return nothing
end

# ϕ-weighted viscoelastic combine for the preconditioner row. Any G·dt that is not a positive
# modulus falls back to the elastic-free combine ϕ·η: a degenerate phase-ratio sample makes the
# shear-modulus lookup 0 or NaN at cells ϕ still marks valid, and the row must stay finite.
# G·dt = Inf (no elasticity) keeps its usual meaning through inv(Inf) = 0.
@inline function ϕ_weighted_harmonic(ϕij, ηij, Gij, dt)
    iszero(ϕij) && return zero(ηij)
    Gdt = Gij * dt
    invGdt = Gdt > 0 ? inv(Gdt) : zero(ηij)
    return ϕij / (inv(ηij) + invGdt)
end

function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, phase_ratios, ϕ::JustRelax.RockRatio, rheology, di, dt, ρgy = nothing)
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
        ϕ,
        rheology,
        dt,
        ρgy,
    )
    return nothing
end

@parallel_indices (i, j) function _Gershgorin_Stokes2D_SchurComplement!(
        Dx, Dy, λmaxVx, λmaxVy, η, ηv, γ_eff, di_center, di_vertex,
        phase_vertex, phase_center, ϕ::JustRelax.RockRatio, rheology, dt, ρgy
    )

    # @inbounds begin
    phase = phase_vertex[i + 1, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[i + 1, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[i, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)


    ηN = ηv[i + 1, j + 1]
    ηS = ηv[i + 1, j]
    ηW = η[i, j]

    γW = γ_eff[i, j] * ϕ.center[i, j]

    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
        if isvalid_vx(ϕ, i + 1, j)
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
            γE = γ_eff[i + 1, j] * ϕ.center[i + 1, j]
            # effective viscoelastic viscosity, ϕ-weighted after the combine
            ηN = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j + 1], ηN, GN, dt)
            ηS = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j], ηS, GS, dt)
            ηW = ϕ_weighted_harmonic(ϕ.center[i, j], ηW, GW, dt)
            ηE = ϕ_weighted_harmonic(ϕ.center[i + 1, j], ηE, GE, dt)

            # Precompute common terms
            ηN_dy = ηN * _dy
            ηS_dy = ηS * _dy
            ηE_dx = ηE * _dx
            ηW_dx = ηW * _dx
            γE_dx = γE * _dx
            γW_dx = γW * _dx

            # compute Gershgorin entries
            Cxx = abs(ηN * _dy2) +
                abs(ηS * _dy2) +
                abs((γE + c43 * ηE) * _dx2) +
                abs((γW + c43 * ηW) * _dx2) +
                abs((ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx)

            Cxy = abs((γE - c23 * ηE + ηN) * _dxdy) +
                abs((γE - c23 * ηE + ηS) * _dxdy) +
                abs((γW + ηN - c23 * ηW) * _dxdy) +
                abs((γW + ηS - c23 * ηW) * _dxdy)

            Dx_ij = (ηN_dy + ηS_dy) * _dy + (γE_dx + γW_dx + c43 * (ηE_dx + ηW_dx)) * _dx
            set_preconditioner!(Dx, λmaxVx, Dx_ij, Cxx + Cxy, i, j)
        else
            Dx[i, j] = one(eltype(Dx))
            λmaxVx[i, j] = one(eltype(λmaxVx))
        end
    end

    # ϕ-weighted viscosity coefficients at surrounding points
    GS = GW # reuse cached value
    phase = phase_vertex[i, j + 1]
    GW = fn_ratio(get_shear_modulus, rheology, phase)
    GE = GN # reuse cached value

    ηS = η[i, j]
    ηW = ηv[i, j + 1]
    ηE = ηv[i + 1, j + 1]
    # bulk-penalty coupling (ϕc² as above); γW already carries its ϕ.center[i, j]
    γS = γW # reuse cached value

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        if isvalid_vy(ϕ, i, j + 1)
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
            γN = γ_eff[i, j + 1] * ϕ.center[i, j + 1]
            # effective viscoelastic viscosity, ϕ-weighted after the combine (see Vx block)
            ηN = ϕ_weighted_harmonic(ϕ.center[i, j + 1], ηN, GN, dt)
            ηS = ϕ_weighted_harmonic(ϕ.center[i, j], ηS, GS, dt)
            ηW = ϕ_weighted_harmonic(ϕ.vertex[i, j + 1], ηW, GW, dt)
            ηE = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j + 1], ηE, GE, dt)

            # Precompute common terms
            ηE_dx = ηE * _dx
            ηW_dx = ηW * _dx
            ηN_dy = ηN * _dy
            ηS_dy = ηS * _dy
            γN_dy = γN * _dy
            γS_dy = γS * _dy

            # Viscous+penalty diagonal augmented by the ϕ-masked FSSA term, as above.
            Dy_visc = (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
            Dy_mag = Dy_visc + abs(fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ))

            # compute Gershgorin entries
            Cyy = abs(ηE * _dx2) +
                abs(ηW * _dx2) +
                abs((γN + c43 * ηN) * _dy2) +
                abs((γS + c43 * ηS) * _dy2) +
                Dy_mag

            Cyx = abs((γN + ηE - c23 * ηN) * _dxdy) +
                abs((γN - c23 * ηN + ηW) * _dxdy) +
                abs((γS + ηE - c23 * ηS) * _dxdy) +
                abs((γS - c23 * ηS + ηW) * _dxdy)

            set_preconditioner!(Dy, λmaxVy, Dy_mag, Cyx + Cyy, i, j)
        else
            Dy[i, j] = one(eltype(Dy))
            λmaxVy[i, j] = one(eltype(λmaxVy))
        end
    end
    # end

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
            βV[i][I...] = @muladd 2 * dτV_ij / (2 + cV_ij * dτV_ij)
            αV[i][I...] = @muladd (2 - cV_ij * dτV_ij) / (2 + cV_ij * dτV_ij)
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
            βV[i][I...] = @muladd 2 * dτV_ij / (2 + cV_ij * dτV_ij)
            αV[i][I...] = @muladd (2 - cV_ij * dτV_ij) / (2 + cV_ij * dτV_ij)
        end
    end
    return nothing
end

# 2D wrapper for update_α_β!
function update_α_β!(dyrel::JustRelax.DYREL)
    return update_α_β!(
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.cVx, dyrel.cVy)
    )
end

# 2D wrapper for update_dτV_α_β!
function update_dτV_α_β!(dyrel::JustRelax.DYREL)
    return update_dτV_α_β!(
        (dyrel.dτVx, dyrel.dτVy),
        (dyrel.βVx, dyrel.βVy),
        (dyrel.αVx, dyrel.αVy),
        (dyrel.cVx, dyrel.cVy),
        (dyrel.λmaxVx, dyrel.λmaxVy),
        dyrel.CFL
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

# # 3D wrapper for update_α_β!
# function update_α_β!(dyrel::JustRelax.DYREL)
#     return update_α_β!(
#         (dyrel.βVx,  dyrel.βVy, dyrel.βVz),
#         (dyrel.αVx,  dyrel.αVy, dyrel.αVz),
#         (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
#         (dyrel.cVx,  dyrel.cVy, dyrel.cVz)
#     )
# end

# # 3D wrapper for update_dτV_α_β!
# function update_dτV_α_β!(dyrel::JustRelax.DYREL)
#     return update_dτV_α_β!(
#         (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
#         (dyrel.βVx, dyrel.βVy, dyrel.βVz),
#         (dyrel.αVx, dyrel.αVy, dyrel.αVz),
#         (dyrel.cVx, dyrel.cVy, dyrel.cVz),
#         (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
#         dyrel.CFL
#     )
# end
