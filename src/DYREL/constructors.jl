"""
    DYREL(ni::NTuple{N, Integer}; ϵ=1e-6, ϵ_vel=1e-6, CFL=0.99, c_fact=0.5, γfact=20.0) where N

Creates a new `DYREL` struct with fields initialized to zero.

# Arguments
- `ni`: Tuple containing the grid dimensions `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D.
- `ϵ`: General convergence tolerance.
- `ϵ_vel`: Velocity convergence tolerance.
- `CFL`: Courant-Friedrichs-Lewy number.
- `c_fact`: Damping scaling factor.
- `γfact`: Scales the Powell-Hestenes penalty against the local viscosity, `γ_num = γfact * η`.
  It is the dominant control on the iteration count, and its optimum depends on both the problem
  and the resolution.
"""
function DYREL(ni::NTuple{2}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    nx, ny = ni
    # penalty parameter
    γ_eff = @zeros(nx, ny)
    # bulk viscosity
    ηb = @zeros(nx, ny)
    # Diagonal preconditioner arrays
    Dx = @zeros(nx - 1, ny)
    Dy = @zeros(nx, ny - 1)
    Dz = @zeros(1, 1)  # dummy for 2D
    # maximum eigenvalue estimates
    λmaxVx = @zeros(nx - 1, ny)
    λmaxVy = @zeros(nx, ny - 1)
    λmaxVz = @zeros(1, 1)  # dummy for 2D
    dVxdτ = @zeros(nx - 1, ny)
    dVydτ = @zeros(nx, ny - 1)
    dVzdτ = @zeros(1, 1)  # dummy for 2D
    dτVx = @zeros(nx - 1, ny)
    dτVy = @zeros(nx, ny - 1)
    dτVz = @zeros(1, 1)  # dummy for 2D
    dVx = @zeros(nx - 1, ny)
    dVy = @zeros(nx, ny - 1)
    dVz = @zeros(1, 1)  # dummy for 2D
    βVx = @zeros(nx - 1, ny)
    βVy = @zeros(nx, ny - 1)
    βVz = @zeros(1, 1)  # dummy for 2D
    cVx = @zeros(nx - 1, ny)
    cVy = @zeros(nx, ny - 1)
    cVz = @zeros(1, 1)  # dummy for 2D
    αVx = @zeros(nx - 1, ny)
    αVy = @zeros(nx, ny - 1)
    αVz = @zeros(1, 1)  # dummy for 2D
    P_num = @zeros(nx, ny)
    Rx0 = @zeros(nx - 1, ny)
    Ry0 = @zeros(nx, ny - 1)
    Rz0 = @zeros(1, 1)  # dummy for 2D

    T = typeof(γ_eff)
    F = typeof(CFL)
    return JustRelax.DYREL{T, F}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0,
        Rz0, CFL, ϵ, ϵ_vel, c_fact, γfact
    )
end

DYREL(nx::Integer, ny::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)

function DYREL(ni::NTuple{3}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    nx, ny, nz = ni
    # penalty parameter
    γ_eff = @zeros(nx, ny, nz)
    # bulk viscosity
    ηb = @zeros(nx, ny, nz)
    # Diagonal preconditioner arrays
    Dx = @zeros(nx - 1, ny, nz)
    Dy = @zeros(nx, ny - 1, nz)
    Dz = @zeros(nx, ny, nz - 1)
    # maximum eigenvalue estimates
    λmaxVx = @zeros(nx - 1, ny, nz)
    λmaxVy = @zeros(nx, ny - 1, nz)
    λmaxVz = @zeros(nx, ny, nz - 1)
    dVxdτ = @zeros(nx - 1, ny, nz)
    dVydτ = @zeros(nx, ny - 1, nz)
    dVzdτ = @zeros(nx, ny, nz - 1)
    dτVx = @zeros(nx - 1, ny, nz)
    dτVy = @zeros(nx, ny - 1, nz)
    dτVz = @zeros(nx, ny, nz - 1)
    dVx = @zeros(nx - 1, ny, nz)
    dVy = @zeros(nx, ny - 1, nz)
    dVz = @zeros(nx, ny, nz - 1)
    βVx = @zeros(nx - 1, ny, nz)
    βVy = @zeros(nx, ny - 1, nz)
    βVz = @zeros(nx, ny, nz - 1)
    cVx = @zeros(nx - 1, ny, nz)
    cVy = @zeros(nx, ny - 1, nz)
    cVz = @zeros(nx, ny, nz - 1)
    αVx = @zeros(nx - 1, ny, nz)
    αVy = @zeros(nx, ny - 1, nz)
    αVz = @zeros(nx, ny, nz - 1)
    P_num = @zeros(nx, ny, nz)
    Rx0 = @zeros(nx - 1, ny, nz)
    Ry0 = @zeros(nx, ny - 1, nz)
    Rz0 = @zeros(nx, ny, nz - 1)

    T = typeof(γ_eff)
    F = typeof(CFL)
    return JustRelax.DYREL{T, F}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0,
        Rz0, CFL, ϵ, ϵ_vel, c_fact, γfact
    )
end

DYREL(nx::Integer, ny::Integer, nz::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny, nz); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)


DYREL(::Type{CPUBackend}, ni::NTuple{N, Integer}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) where {N} = DYREL(ni; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
DYREL(::Type{CPUBackend}, nx::Integer, ny::Integer, nz::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny, nz); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)

function DYREL(::Type{CPUBackend}, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    return DYREL(stokes, rheology, phase_ratios, di, dt; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
end

function DYREL(::Type{CPUBackend}, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ::JustRelax.RockRatio, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    return DYREL(stokes, rheology, phase_ratios, ϕ, di, dt; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
end


"""
    DYREL(stokes, rheology, phase_ratios, di, dt; ϵ=1e-6, ϵ_vel=1e-6, CFL=0.99, c_fact=0.5, γfact=20.0)

Constructs and initializes a `DYREL` object based on existing Stokes fields.

This function:
1. Allocates zero-initialized arrays using grid dimensions from `stokes`.
2. Computes initial bulk viscosity and penalty parameters.
3. Computes Gershgorin estimates for eigenvalues and preconditioners.
4. Updates damping coefficients.

# Arguments
- `stokes`: `JustRelax.StokesArrays` struct.
- `rheology`: Material properties.
- `phase_ratios`: Phase fraction information.
- `di`: Grid spacing tuple.
- `dt`: Time step.
- `γfact`: Factor for penalty parameter calculation (default: 20.0).
"""
function DYREL(stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)

    dyrel = DYREL(size(stokes.P); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
    DYREL!(dyrel, stokes, rheology, phase_ratios, di, dt; CFL = CFL, γfact = γfact)

    return dyrel
end


"""
    DYREL(stokes, rheology, phase_ratios, ϕ, di, dt; ϵ=1e-6, ϵ_vel=1e-6, CFL=0.99, c_fact=0.5, γfact=20.0)

Variational counterpart of [`DYREL`](@ref), for solves masked by the `RockRatio` `ϕ`.

`ηb` and the Gershgorin bounds are built with the same ϕ masking the variational solver applies
each time step. The numerical Powell-Hestenes penalty `γ_eff` itself remains unweighted: its
momentum gradient already carries the variational cell weight, so weighting `γ_eff` here would
apply `ϕ²` to the augmented-Lagrangian block.

# Arguments
- `stokes`: `JustRelax.StokesArrays` struct.
- `rheology`: Material properties.
- `phase_ratios`: Phase fraction information.
- `ϕ`: `JustRelax.RockRatio` marking the valid (rock) degrees of freedom.
- `di`: Grid spacing tuple.
- `dt`: Time step.
- `γfact`: Factor for penalty parameter calculation (default: 20.0).
"""
function DYREL(stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ::JustRelax.RockRatio, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)

    dyrel = DYREL(size(stokes.P); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
    DYREL!(dyrel, stokes, rheology, phase_ratios, ϕ, di, dt; CFL = CFL, γfact = γfact)

    return dyrel
end


"""
    DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; CFL=dyrel.CFL, γfact=dyrel.γfact)

Updates the fields of the `DYREL` struct in-place for the current time step.

This function recomputes:
- Bulk viscosity and penalty parameter `γ_eff`.
- Gershgorin estimates for eigenvalues and preconditioners.
- Damping coefficients.

`CFL` and `γfact` default to the values stored in `dyrel` at construction time, so a
custom `γfact`/`CFL` passed to the `DYREL(...)` constructor is respected on every
subsequent in-place update rather than being silently reset to the keyword defaults
below.

# Arguments
- `dyrel`: `JustRelax.DYREL` struct to modify.
- `stokes`: `JustRelax.StokesArrays` containing current simulation state.
- `rheology`, `phase_ratios`: Material properties.
- `di`: Grid spacing.
- `dt`: Current time step.
- `CFL`: Courant number (default: `dyrel.CFL`).
- `γfact`: Penalty factor (default: `dyrel.γfact`).

Returns `nothing`.
"""
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt, ρg_v = nothing; CFL = dyrel.CFL, γfact = dyrel.γfact)
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt, di.center, ρg_v)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt, ρg_v)

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

    return nothing
end

# variational version
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ::JustRelax.RockRatio, di, dt, ρg_v = nothing; CFL = dyrel.CFL, γfact = dyrel.γfact)
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt, di.center, ρg_v)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners (ϕ-aware)
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, ϕ, rheology, di, dt, ρg_v)

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

    return nothing
end

# Mean over the non-infinite entries, reduced globally so an infinite-viscosity fallback is
# independent of the MPI decomposition. A NaN entry still propagates to the result, as it must.
# `Array` gets sum and count in one traversal; every other array type — GPU arrays among them —
# takes two whole-array reductions, which is what a device can execute.
@inline function noninf_stats(A::Array)
    s, n = zero(eltype(A)), 0
    for a in A
        if !isinf(a)
            s += a
            n += 1
        end
    end
    return s, n
end
@inline noninf_stats(A) = (sum(a -> isinf(a) ? zero(a) : a, A), sum(a -> isinf(a) ? 0 : 1, A))
@inline function noninf_mean(A)
    local_sum, local_count = noninf_stats(A)
    global_sum, global_count = if MPI.Initialized() && !MPI.Finalized()
        (
            MPI.Allreduce(local_sum, MPI.SUM, MPI.COMM_WORLD),
            MPI.Allreduce(local_count, MPI.SUM, MPI.COMM_WORLD),
        )
    else
        local_sum, local_count
    end
    iszero(global_count) && throw(ArgumentError("cannot scale the DYREL penalty: viscosity is infinite everywhere"))
    return global_sum / global_count
end

# Local viscosity, so that γ_eff/η stays O(γfact) everywhere: a domain-mean scaling
# over-penalizes the soft cells (dynamic-relaxation cost grows with γ/η) and under-penalizes
# the stiff ones (Powell-Hestenes residual drops by η/(γ+η) per pass). Mean is the fallback
# where the local viscosity is infinite.
@inline penalty_viscosity(η_local, η_mean) = isinf(η_local) ? η_mean : η_local

# Lower bound on γ_eff imposed by the free-surface stabilization term: `|Δ(ρg)|·dt·dy/2` is
# the break-even at which the vertical-momentum row is viscous- rather than buoyancy-dominated
# (its FSSA diagonal is S = |∂(ρg)/∂y|·dt against a preconditioner diagonal ≈ (2γ_eff +
# 8/3·η)/dy²; where S ≫ D the surface row decouples and relaxes far slower than the bulk).
# Δ(ρg) spans both vertical faces so the rows on either side of a density jump are covered.
# Returns a Bool `false` (numeric zero) when no buoyancy field is supplied, so the FSSA-off
# path is byte-identical.
@inline fssa_penalty_floor(::Nothing, di_center, dt, I::Vararg{Integer, N}) where {N} = false

Base.@propagate_inbounds @inline function fssa_penalty_floor(ρg_v, di_center, dt, i, j)
    ρg_c = ρg_v[i, j]
    Δρg = max(
        abs(ρg_v[i, min(j + 1, size(ρg_v, 2))] - ρg_c),
        abs(ρg_c - ρg_v[i, max(j - 1, 1)]),
    )
    return Δρg * dt * clamped_dy(di_center[2], j) / 2
end

# Center-to-center spacing has one entry fewer than there are cells — no gap past the last
# center — so the last row reuses the last gap, matching the clamped ρg_v lookups above.
@inline clamped_dy(dy::Number, ::Integer) = dy
@inline clamped_dy(dy::AbstractVector, j::Integer) = dy[min(j, lastindex(dy))]

"""
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt, di_center=nothing, ρg_v=nothing)

Computes the bulk viscosity `ηb` and the effective penalty parameter `γ_eff`.

1. **Bulk Viscosity (`ηb`)**: `ηb = Kb * dt`, infinite for an incompressible phase.

2. **Penalty Parameter (`γ_eff`)**: harmonic combination of a numerical (`γ_num`) and a
   physical (`γ_phy`) penalty term.
   - `γ_num = γfact * η` (local viscosity; falls back to `η_mean` where `η` is infinite)
   - `γ_phy = Kb * dt` (or `γ_num` where `Kb` is infinite, giving `γ_eff = γ_num / 2`)
   - `γ_eff = (γ_phy * γ_num) / (γ_phy + γ_num)`, floored at `fssa_penalty_floor` when a
     vertical buoyancy field is supplied.

# Arguments
- `dyrel`: `JustRelax.DYREL` struct to update.
- `stokes`: `JustRelax.StokesArrays`.
- `rheology`: Material properties.
- `phase_ratios`: Phase fraction information.
- `γfact`: Numerical factor for penalty parameter.
- `dt`: Time step.
- `di_center`: Cell-center grid spacing; required for the free-surface floor.
- `ρg_v`: Vertical buoyancy `ρg`; `nothing` disables the free-surface floor.

This function parallelizes the computation across grid cells.
"""
function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt, di_center = nothing, ρg_v = nothing)
    ni = size(stokes.P)
    @parallel (@idx ni) compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, stokes.viscosity.η, noninf_mean(stokes.viscosity.η), γfact, dt, di_center, ρg_v)
    return nothing
end


@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, η, η_mean, γfact, dt, di_center, ρg_v)

    # bulk viscosity
    ratios = @inbounds @cell phase_ratios_center[I...]
    Kbdt = fn_ratio(get_bulk_modulus, rheology, ratios) * dt
    ηb[I...] = Kbdt

    γ_num = γfact * penalty_viscosity(η[I...], η_mean)
    γ_phy = isinf(Kbdt) ? γ_num : Kbdt
    # Local scaling alone leaves γ_eff set by the (soft) air viscosity at a sticky-air free
    # surface, far below the FSSA diagonal; floor it so the two stay commensurate.
    γ_eff[I...] = max(γ_phy * γ_num / (γ_phy + γ_num), fssa_penalty_floor(ρg_v, di_center, dt, I...))

    return nothing
end


# variational version

"""
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt, di_center=nothing, ρg_v=nothing)

Variational counterpart of the method above, masked by the rock ratio `ϕ`.

Cells that fail `isvalid_c` get `ηb = γ_eff = 0`. A non-positive or `NaN` bulk modulus is
treated as the incompressible limit, keeping both quantities finite — and leaving `ηb = Inf`
whatever `ϕ` is, so the weighting below acts on compressible phases only.

The two weightings carry different meanings, and neither follows from the other:

  - `ηb = ϕ·K·dt` is an *effective bulk modulus*. The residual `-∇V - (P - P0)/ηb` measures the
    divergence of the whole cell against the volumetric response of the rock alone, and a cell
    that is only `ϕ` rock resists a volume change proportionally less because the rest of it is
    void. The divergence term is correspondingly left unweighted: it is the cell-average
    divergence, void included.

  - `γ_eff = γ` is the Powell-Hestenes *penalty* on the retained constraint `RP = 0`.
    Downstream, the momentum stencil differences `θc = γ_eff·RP + ΔPψ` through
    `ϕ.center`-weighted neighbours, supplying the single variational volume weight required by
    the discrete pressure gradient. Weighting `γ_eff` too would spuriously produce `ϕ²`.
"""
function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ::JustRelax.RockRatio, γfact, dt, di_center = nothing, ρg_v = nothing)
    ni = size(stokes.P)
    η_mean = noninf_mean(stokes.viscosity.η)
    @parallel (@idx ni) compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, ϕ, stokes.viscosity.η, η_mean, γfact, dt, di_center, ρg_v)
    return nothing
end

@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, ϕ::JustRelax.RockRatio, η, η_mean, γfact, dt, di_center, ρg_v)

    if isvalid_c(ϕ, I...)
        # bulk viscosity: ϕ·K·dt is the effective bulk modulus of a cell that is only ϕ rock, so
        # it pairs with an unweighted divergence (see the docstring). A non-positive or NaN bulk
        # modulus falls back to the incompressible limit so ηb/γ_eff stay finite (as
        # `ϕ_weighted_harmonic` does in Gershgorin.jl).
        ratios = @cell phase_ratios_center[I...]
        Kbdt = fn_ratio(get_bulk_modulus, rheology, ratios) * dt
        Kbdt = Kbdt > 0 ? Kbdt : oftype(Kbdt, Inf)
        ηb[I...] = Kbdt * ϕ.center[I...]

        # Numerical penalty on the retained RP = 0 constraint. Do not weight it by ϕ here: the
        # variational momentum gradient applies ϕ.center once downstream.
        γ_num = γfact * penalty_viscosity(η[I...], η_mean)
        γ_phy = isinf(Kbdt) ? γ_num : Kbdt
        γ_eff[I...] = max(γ_phy * γ_num / (γ_phy + γ_num), fssa_penalty_floor(ρg_v, di_center, dt, I...))
    else
        ηb[I...] = 0.0e0
        γ_eff[I...] = 0.0e0
    end

    return nothing
end
