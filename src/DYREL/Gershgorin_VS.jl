# The variational free-surface stabilization keeps only the stabilizing part of ∂(ρg)/∂y, where
# density decreases upward (the rock/air interface, stable stratification). Where density
# increases upward the full term would anticipate Rayleigh–Taylor growth over `dt`; once `dt`
# exceeds that growth time it makes the momentum operator indefinite and the pseudo-transient
# iterations diverge. Left out, such growth is resolved by the advective time step instead.
@inline stabilizing_∂ρg∂y(∂ρg∂y) = min(∂ρg∂y, zero(∂ρg∂y))

# Diagonal contribution of the free-surface stabilization buoyancy term (Kaus et al.,
# 2010) to the Vy operator: -∂(ρg)/∂y·θ·dt, with θ = 1 and ∂(ρg)/∂y clipped by
# `stabilizing_∂ρg∂y`, so it never reduces the diagonal. The residual kernels
# (`compute_PH_residual_V!`/`compute_DR_residual_update_V!`) add Vy·∂(ρg)/∂y·θ·dt to Ry, and
# Ry = b - A·V, so the term enters A's diagonal with the opposite sign — the same convention as
# `free_surface_diagonal`. Passing a `RockRatio` selects the ϕ-masked ρg sampling those masked
# kernels use, so the preconditioner describes the same stabilized operator. Returns a Bool
# `false` (numeric zero) when no buoyancy field is supplied, so construction-time estimates and
# the FSSA-off path are byte-identical to the plain viscous diagonal.
@inline fssa_diagonal_y(::Nothing, i, j, _dy, dt, ϕ = nothing) = false
Base.@propagate_inbounds @inline function fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ = nothing)
    # the launch spans every center row, so the north neighbour of the top row does not exist
    j == lastindex(ρgy, 2) && return zero(eltype(ρgy))
    return -stabilizing_∂ρg∂y(_d_ya_ρg(ρgy, ϕ, _dy, i, j)) * dt
end

Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ::Nothing, _dy, i, j) = _d_ya(ρgy, _dy, i, j)
Base.@propagate_inbounds @inline _d_ya_ρg(ρgy, ϕ::JustRelax.RockRatio, _dy, i, j) = _d_ya(ρgy, ϕ.center, _dy, i, j)

# Store a preconditioner diagonal and its eigenvalue bound. The viscosity and penalty
# coefficients passed as `diagonal` already contain the volume-fraction weighting of the
# variational residual. Multiplying the complete diagonal by the face fraction here would
# apply that weighting a second time (`ϕ²`) for a uniform cut cell.
#
# A diagonal of exactly zero is a legitimate mask-boundary row that `isvalid_*` accepts but whose
# neighbouring stress weights all vanish; it carries no coupling, so the identity is the right
# preconditioner for it. A negative or non-finite diagonal is not legitimate — it means a
# degenerate viscosity sample — and is propagated as `NaN`, the same signal
# `free_surface_pseudotime` raises, rather than silently replaced by 1.
Base.@propagate_inbounds @inline function set_preconditioner!(D, λmaxV, diagonal, row_sum, i, j)
    if diagonal > zero(diagonal)
        D[i, j] = diagonal
        λmaxV[i, j] = row_sum / diagonal
    elseif iszero(diagonal)
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

# Vertex viscosity seen by the variational operator. `compute_stress_DYREL!` builds the
# vertex stress from the harmonic mean of the four surrounding center viscosities, clamped
# at the domain border, so the Gershgorin bound must sample that same combination for the
# preconditioner to describe the operator being iterated.
Base.@propagate_inbounds @inline function η_vertex(η, ni, periodic, i, j)
    return harm_clamped(η, clamped_indices(ni, periodic, i, j)...)
end

function Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, η, γ_eff, phase_ratios, ϕ::JustRelax.RockRatio, rheology, di, dt, ρgy = nothing)
    ni = size(η)
    # Fixed 256-thread blocks avoid heuristic rounding above the GPU kernel limit.
    @parallel (@idx ni) (cld(ni[1], 32), cld(ni[2], 8), 1) (32, 8, 1) _Gershgorin_Stokes2D_SchurComplement!(
        Dx,
        Dy,
        λmaxVx,
        λmaxVy,
        η,
        γ_eff,
        di.center,
        di.vertex,
        di.velocity,
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
        Dx, Dy, λmaxVx, λmaxVy, η, γ_eff, di_center, di_vertex, di_velocity,
        phase_vertex, phase_center, ϕ::JustRelax.RockRatio, rheology, dt, ρgy
    )

    ni = size(η)
    # A direction is periodic exactly when its momentum block carries a row per cell rather than
    # one per interior face, so the shapes of `Dx`/`Dy` are the flags -- no extra argument.
    periodic = (size(Dx, 1) == ni[1], size(Dy, 2) == ni[2])

    # @inbounds begin
    phase = phase_vertex[i + 1, j + 1]
    GN = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_vertex[i + 1, j]
    GS = fn_ratio(get_shear_modulus, rheology, phase)
    phase = phase_center[i, j]
    GW = fn_ratio(get_shear_modulus, rheology, phase)


    ηN = η_vertex(η, ni, periodic, i + 1, j + 1)
    ηS = η_vertex(η, ni, periodic, i + 1, j)
    ηW = η[i, j]

    # The Schur term carries one fraction from RP and one from its masked gradient.
    # Keep both even though γ_eff already compensates one in the pressure step.
    γW = γ_eff[i, j] * ϕ.center[i, j]^2

    if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
        if isvalid_vx(ϕ, i + 1, j)
            # Hoist common parameters
            dx = @dx(di_center, i)
            dy = @dy(di_vertex, j)
            _dx = inv(dx)
            _dy = inv(dy)
            _dxdy = _dx * _dy
            c43 = 4 / 3
            c23 = 2 / 3

            # `Dx` reaches `i = ni[1]` only when x is periodic; that row is the seam face, whose
            # eastern cell is the first one. The vertex reads need no wrap -- `ni[1] + 1` is
            # already the seam plane of the vertex arrays.
            iE = wrap_next(i, ni[1])
            phase = phase_center[iE, j]
            GE = fn_ratio(get_shear_modulus, rheology, phase)
            ηE = η[iE, j]
            γE = γ_eff[iE, j] * ϕ.center[iE, j]^2
            # effective viscoelastic viscosity, ϕ-weighted after the combine
            ηN = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j + 1], ηN, GN, dt)
            ηS = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j], ηS, GS, dt)
            ηW = ϕ_weighted_harmonic(ϕ.center[i, j], ηW, GW, dt)
            ηE = ϕ_weighted_harmonic(ϕ.center[iE, j], ηE, GE, dt)

            # Differentiate with the spacing of each source cell/vertex, then that
            # of the momentum row; these coincide only on a uniform grid.
            aN = ηN / (@dy di_velocity[1] j + 1) * _dy
            aS = ηS / (@dy di_velocity[1] j) * _dy
            aE = (γE + c43 * ηE) / (@dx di_vertex iE) * _dx
            aW = (γW + c43 * ηW) / (@dx di_vertex i) * _dx
            Dx_ij = aN + aS + aE + aW
            Cxx = abs(aN) + abs(aS) + abs(aE) + abs(aW) + abs(Dx_ij)

            Cxy = abs((γE - c23 * ηE + ηN) * _dxdy) +
                abs((γE - c23 * ηE + ηS) * _dxdy) +
                abs((γW + ηN - c23 * ηW) * _dxdy) +
                abs((γW + ηS - c23 * ηW) * _dxdy)

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
    ηW = η_vertex(η, ni, periodic, i, j + 1)
    ηE = η_vertex(η, ni, periodic, i + 1, j + 1)
    # Powell-Hestenes penalty coupling; γW already carries the ϕ.center[i, j] weights.
    γS = γW # reuse cached value

    if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
        if isvalid_vy(ϕ, i, j + 1)
            # Hoist common parameters
            dx = @dx(di_vertex, i)
            dy = @dy(di_center, j)
            _dx = inv(dx)
            _dy = inv(dy)
            _dxdy = _dx * _dy
            c43 = 4 / 3
            c23 = 2 / 3

            # Counterpart of `iE` in the Vx block: the northern cell of the y seam face is the
            # first one.
            jN = wrap_next(j, ni[2])
            phase = phase_center[i, jN]
            GN = fn_ratio(get_shear_modulus, rheology, phase)

            ηN = η[i, jN]
            γN = γ_eff[i, jN] * ϕ.center[i, jN]^2
            # effective viscoelastic viscosity, ϕ-weighted after the combine (see Vx block)
            ηN = ϕ_weighted_harmonic(ϕ.center[i, jN], ηN, GN, dt)
            ηS = ϕ_weighted_harmonic(ϕ.center[i, j], ηS, GS, dt)
            ηW = ϕ_weighted_harmonic(ϕ.vertex[i, j + 1], ηW, GW, dt)
            ηE = ϕ_weighted_harmonic(ϕ.vertex[i + 1, j + 1], ηE, GE, dt)

            aE = ηE / (@dx di_velocity[2] i + 1) * _dx
            aW = ηW / (@dx di_velocity[2] i) * _dx
            aN = (γN + c43 * ηN) / (@dy di_vertex jN) * _dy
            aS = (γS + c43 * ηS) / (@dy di_vertex j) * _dy
            Dy_visc = aE + aW + aN + aS
            Dy_ij = Dy_visc + fssa_diagonal_y(ρgy, i, j, _dy, dt, ϕ)
            Cyy = abs(aE) + abs(aW) + abs(aN) + abs(aS) + Dy_ij

            Cyx = abs((γN + ηE - c23 * ηN) * _dxdy) +
                abs((γN - c23 * ηN + ηW) * _dxdy) +
                abs((γS + ηE - c23 * ηS) * _dxdy) +
                abs((γS - c23 * ηS + ηW) * _dxdy)

            set_preconditioner!(Dy, λmaxVy, Dy_ij, Cyx + Cyy, i, j)
        else
            Dy[i, j] = one(eltype(Dy))
            λmaxVy[i, j] = one(eltype(λmaxVy))
        end
    end
    # end

    return nothing
end

function Gershgorin_variational!(::Val{2}, dyrel, stokes, phases, ϕ, rheology, di, dt, ρg = nothing)
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, dyrel.γ_eff, phases, ϕ, rheology, di, dt, ρg)
    # At a rank seam `η_vertex` clamps the outermost vertex onto the local cells, so the face
    # rows next to it get a one-sided bound while the neighbor computes the same rows from the
    # full stencil. Take them from the neighbor; otherwise the two copies of a shared face relax
    # with different pseudo-time steps. Overlap-1 directions carry no halo and are skipped. A
    # single rank has no seam, and its arrays need not match the global grid it initialized.
    if ImplicitGlobalGrid.grid_is_initialized() && ImplicitGlobalGrid.global_grid().nprocs > 1 && !any(periodic_dims(stokes))
        update_halo!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy)
    end
    return nothing
end
