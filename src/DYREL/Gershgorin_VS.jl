# Diagonal contribution of the free-surface stabilization buoyancy term (Kaus et al.,
# 2010) to the Vy operator: -∂(ρg)/∂y·θ·dt, with θ = 1. The residual kernels
# (`compute_PH_residual_V!`/`compute_DR_residual_update_V!`) add Vy·∂(ρg)/∂y·θ·dt to Ry, and
# Ry = b - A·V, so the term enters A's diagonal with the opposite sign — the same convention as
# `free_surface_diagonal`. It is signed, not a magnitude: where density increases upward it
# reduces the diagonal. Passing a `RockRatio` selects the ϕ-masked ρg sampling those masked
# kernels use, so the preconditioner describes the same stabilized operator. Returns a Bool
# `false` (numeric zero) when no buoyancy field is supplied, so construction-time estimates and
# the FSSA-off path are byte-identical to the plain viscous diagonal.
@inline fssa_diagonal_y(::Nothing, i, j, _dy, dt, ϕ = nothing) = false
Base.@propagate_inbounds @inline function fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ = nothing)
    # the launch spans every center row, so the north neighbour of the top row does not exist
    j == lastindex(ρgy, 2) && return zero(eltype(ρgy))
    return -_d_ya_ρg(ρgy, ϕ, _dy, i, j) * dt
end

Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ::Nothing, _dy, i, j) = _d_ya(ρgy, _dy, i, j)
Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ϕ::JustRelax.RockRatio, _dy, i, j) = _d_ya(ρgy, ϕ.center, _dy, i, j)

# Store a preconditioner diagonal and its eigenvalue bound. The bounded face mass is the outer
# `W_L^u` factor of the variational velocity preconditioner. The physical row sum is not rescaled,
# so sliver faces correctly increase the preconditioned eigenvalue bound.
#
# A diagonal of exactly zero is a legitimate mask-boundary row that `isvalid_*` accepts but whose
# neighbouring stress weights all vanish; it carries no coupling, so the identity is the right
# preconditioner for it. A negative or non-finite diagonal is not legitimate — it means either a
# degenerate viscosity sample or an FSSA term large enough to invert the row — and is propagated as
# `NaN`, the same signal `free_surface_pseudotime` raises, rather than silently replaced by 1.
Base.@propagate_inbounds @inline function set_preconditioner!(D, λmaxV, diagonal, row_sum, face_fraction, i, j)
    face_mass = variational_face_mass(face_fraction)
    weighted_diagonal = face_mass * diagonal
    if weighted_diagonal > zero(weighted_diagonal)
        D[i, j] = weighted_diagonal
        λmaxV[i, j] = row_sum / weighted_diagonal
    elseif iszero(weighted_diagonal)
        D[i, j] = one(eltype(D))
        λmaxV[i, j] = one(eltype(λmaxV))
    else
        D[i, j] = convert(eltype(D), NaN)
        λmaxV[i, j] = convert(eltype(λmaxV), NaN)
    end
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
            set_preconditioner!(Dx, λmaxVx, Dx_ij, Cxx + Cxy, ϕ.Vx[i + 1, j], i, j)
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
    # Powell-Hestenes penalty coupling; γW already carries the single variational
    # ϕ.center[i, j] weight applied by the momentum gradient.
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

            # Viscous+penalty diagonal augmented by the signed ϕ-masked FSSA term, as above.
            Dy_visc = (γN_dy + γS_dy + c43 * (ηN_dy + ηS_dy)) * _dy + (ηE_dx + ηW_dx) * _dx
            Dy_ij = Dy_visc + fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ)

            # compute Gershgorin entries
            Cyy = abs(ηE * _dx2) +
                abs(ηW * _dx2) +
                abs((γN + c43 * ηN) * _dy2) +
                abs((γS + c43 * ηS) * _dy2) +
                Dy_ij

            Cyx = abs((γN + ηE - c23 * ηN) * _dxdy) +
                abs((γN - c23 * ηN + ηW) * _dxdy) +
                abs((γS + ηE - c23 * ηS) * _dxdy) +
                abs((γS - c23 * ηS + ηW) * _dxdy)

            set_preconditioner!(Dy, λmaxVy, Dy_ij, Cyx + Cyy, ϕ.Vy[i, j + 1], i, j)
        else
            Dy[i, j] = one(eltype(Dy))
            λmaxVy[i, j] = one(eltype(λmaxVy))
        end
    end
    # end

    return nothing
end
