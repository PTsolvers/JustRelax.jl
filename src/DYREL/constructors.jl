"""
    DYREL(ni::NTuple{N, Integer}[, periodic]; ϵ=1e-6, ϵ_vel=1e-6, CFL=0.99, c_fact=0.5, γfact=20.0) where N

Creates a new `DYREL` struct with fields initialized to zero.

# Arguments
- `ni`: Tuple containing the grid dimensions `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D.
- `periodic`: `N`-tuple marking the periodic directions, which each carry one extra momentum row
  (see [`momentum_rows`](@ref)). Defaults to all-`false`. The `StokesArrays` method below reads it
  off the containers instead, so the two cannot disagree.

# Keyword arguments
- `ϵ`: General convergence tolerance. Default: `1.0e-6`.
- `ϵ_vel`: Velocity convergence tolerance. Default: `1.0e-6`.
- `CFL`: Courant-Friedrichs-Lewy number. Default: `0.99`.
- `c_fact`: Damping scaling factor. Default: `0.5`.
- `γfact`: Penalty scaling factor. Default: `20.0`.
"""
function DYREL(ni::NTuple{2}, periodic::NTuple{2, Bool} = (false, false); ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    nx, ny = ni
    # Every per-face array is indexed by the momentum row it belongs to, so they all share the
    # residual's shape -- one row longer in a periodic direction, which carries the seam face.
    nVx = momentum_rows(ni, periodic, 1)
    nVy = momentum_rows(ni, periodic, 2)
    # penalty parameter
    γ_eff = @zeros(nx, ny)
    # bulk viscosity
    ηb = @zeros(nx, ny)
    # Diagonal preconditioner arrays
    Dx = @zeros(nVx...)
    Dy = @zeros(nVy...)
    Dz = @zeros(1, 1)  # dummy for 2D
    # maximum eigenvalue estimates
    λmaxVx = @zeros(nVx...)
    λmaxVy = @zeros(nVy...)
    λmaxVz = @zeros(1, 1)  # dummy for 2D
    dVxdτ = @zeros(nVx...)
    dVydτ = @zeros(nVy...)
    dVzdτ = @zeros(1, 1)  # dummy for 2D
    dτVx = @zeros(nVx...)
    dτVy = @zeros(nVy...)
    dτVz = @zeros(1, 1)  # dummy for 2D
    dVx = @zeros(nVx...)
    dVy = @zeros(nVy...)
    dVz = @zeros(1, 1)  # dummy for 2D
    βVx = @zeros(nVx...)
    βVy = @zeros(nVy...)
    βVz = @zeros(1, 1)  # dummy for 2D
    cVx = @zeros(nVx...)
    cVy = @zeros(nVy...)
    cVz = @zeros(1, 1)  # dummy for 2D
    αVx = @zeros(nVx...)
    αVy = @zeros(nVy...)
    αVz = @zeros(1, 1)  # dummy for 2D
    P_num = @zeros(nx, ny)
    Rx0 = @zeros(nVx...)
    Ry0 = @zeros(nVy...)
    Rz0 = @zeros(1, 1)  # dummy for 2D

    T = typeof(γ_eff)
    F = typeof(CFL)
    return JustRelax.DYREL{T, F}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0,
        Rz0, CFL, γfact, ϵ, ϵ_vel, c_fact
    )
end

DYREL(nx::Integer, ny::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)

function DYREL(ni::NTuple{3}, periodic::NTuple{3, Bool} = (false, false, false); ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    nx, ny, nz = ni
    # Every per-face array is indexed by the momentum row it belongs to, so they all share the
    # residual's shape -- one row longer in a periodic direction, which carries the seam face.
    nVx = momentum_rows(ni, periodic, 1)
    nVy = momentum_rows(ni, periodic, 2)
    nVz = momentum_rows(ni, periodic, 3)
    # penalty parameter
    γ_eff = @zeros(nx, ny, nz)
    # bulk viscosity
    ηb = @zeros(nx, ny, nz)
    # Diagonal preconditioner arrays
    Dx = @zeros(nVx...)
    Dy = @zeros(nVy...)
    Dz = @zeros(nVz...)
    # maximum eigenvalue estimates
    λmaxVx = @zeros(nVx...)
    λmaxVy = @zeros(nVy...)
    λmaxVz = @zeros(nVz...)
    dVxdτ = @zeros(nVx...)
    dVydτ = @zeros(nVy...)
    dVzdτ = @zeros(nVz...)
    dτVx = @zeros(nVx...)
    dτVy = @zeros(nVy...)
    dτVz = @zeros(nVz...)
    dVx = @zeros(nVx...)
    dVy = @zeros(nVy...)
    dVz = @zeros(nVz...)
    βVx = @zeros(nVx...)
    βVy = @zeros(nVy...)
    βVz = @zeros(nVz...)
    cVx = @zeros(nVx...)
    cVy = @zeros(nVy...)
    cVz = @zeros(nVz...)
    αVx = @zeros(nVx...)
    αVy = @zeros(nVy...)
    αVz = @zeros(nVz...)
    P_num = @zeros(nx, ny, nz)
    Rx0 = @zeros(nVx...)
    Ry0 = @zeros(nVy...)
    Rz0 = @zeros(nVz...)

    T = typeof(γ_eff)
    F = typeof(CFL)
    return JustRelax.DYREL{T, F}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0,
        Rz0, CFL, γfact, ϵ, ϵ_vel, c_fact
    )
end

DYREL(nx::Integer, ny::Integer, nz::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny, nz); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)


DYREL(::Type{CPUBackend}, ni::NTuple{N, Integer}, periodic::NTuple{N, Bool} = ntuple(_ -> false, Val(N)); ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) where {N} = DYREL(ni, periodic; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
DYREL(::Type{CPUBackend}, nx::Integer, ny::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0) = DYREL((nx, ny); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
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

# Keyword arguments
- `ϵ`: General convergence tolerance. Default: `1.0e-6`.
- `ϵ_vel`: Velocity convergence tolerance. Default: `1.0e-6`.
- `CFL`: Courant-Friedrichs-Lewy number. Default: `0.99`.
- `c_fact`: Damping scaling factor. Default: `0.5`.
- `γfact`: Factor for the penalty parameter calculation. Default: `20.0`.
"""
function DYREL(stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)

    ni = size(stokes.P)
    dim = Val(length(ni))

    # instantiate DYREL object
    dyrel = DYREL(ni, periodic_dims(stokes); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes_SchurComplement!(dim, dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return dyrel
end

function DYREL(stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ::JustRelax.RockRatio, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    dyrel = DYREL(size(stokes.P), periodic_dims(stokes); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
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

# Arguments
- `dyrel`: `JustRelax.DYREL` struct to modify.
- `stokes`: `JustRelax.StokesArrays` containing current simulation state.
- `rheology`, `phase_ratios`: Material properties.
- `di`: Grid spacing.
- `dt`: Current time step.
- `CFL`: Courant number (default: the value stored in `dyrel`).
- `γfact`: Penalty factor (default: the value stored in `dyrel`).

Returns `nothing`.
"""
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; CFL = dyrel.CFL, γfact = dyrel.γfact)
    dim = Val(ndims(stokes.P))

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes_SchurComplement!(dim, dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return nothing
end

# variational version
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ, di, dt, ρgy = nothing; CFL = dyrel.CFL, γfact = dyrel.γfact)
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, dyrel.γ_eff, phase_ratios, ϕ, rheology, di, dt, ρgy)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return nothing
end


"""
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

Computes the bulk viscosity `ηb` and the effective penalty parameter `γ_eff`.

1. **Bulk Viscosity (`ηb`)**: Computed based on the bulk modulus of the material phases.
   - If `Kb` is infinite (incompressible), `ηb` defaults to `γfact * η_mean`.
   - Otherwise `ηb = Kb * dt`.

2. **Penalty Parameter (`γ_eff`)**: A combination of numerical (`γ_num`) and physical (`γ_phy`) penalty terms.
   - `γ_num = γfact * η_mean`, with `η_mean` the mean of the finite viscosities
   - `γ_phy = Kb * dt` (or `γ_num` where `Kb` is infinite)
   - `γ_eff = (γ_phy * γ_num) / (γ_phy + γ_num)`

# Arguments
- `dyrel`: `JustRelax.DYREL` struct to update.
- `stokes`: `JustRelax.StokesArrays`.
- `rheology`: Material properties.
- `phase_ratios`: Phase fraction information.
- `γfact`: Numerical factor for penalty parameter (default: 20.0).
- `dt`: Time step.

This function parallelizes the computation across grid cells.
"""
function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)
    ni = size(stokes.P)
    @parallel (@idx ni) compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, mean(stokes.viscosity.η[.!isinf.(stokes.viscosity.η)]), γfact, dt)
    return nothing
end


@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, η_mean, γfact, dt)

    # bulk viscosity
    ratios = @inbounds @cell phase_ratios_center[I...]
    Kbdt = fn_ratio(get_bulk_modulus, rheology, ratios) * dt
    ηb[I...] = Kbdt

    # penalty parameter: scaled by a single global viscosity. A penalty scaled by the local
    # viscosity cannot relax the pressure of a weak body enclosed by strong material (e.g. a
    # thermally contracting magma chamber): that pressure mode is resisted by the strong
    # surroundings, so a step sized by the weak viscosity barely reduces it.
    γ_num = γfact * η_mean
    γ_phy = isinf(Kbdt) ? γ_num : Kbdt
    γ_eff[I...] = γ_phy * γ_num / (γ_phy + γ_num)

    return nothing
end


# variational version

function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)
    ni = size(stokes.P)
    @parallel (@idx ni) compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, stokes.viscosity.η, ϕ, mean(stokes.viscosity.η[.!isinf.(stokes.viscosity.η)]), γfact, dt)
    return nothing
end

# `ηb` holds the same material quantity as the full-volume solver: the rock fraction
# reaches the continuity equation through the assembled residual (see
# `variational_continuity_residual`), not through its coefficients.
#
# `γ_eff` is not a coefficient of that equation but the step length of the pressure
# update `P += γ_eff * RP`, so it must undo the weight `RP` carries. The Schur
# complement of a cut cell is proportional to `ϕ.center` — the row is weighted by it
# while the momentum rows it drives are normalized by their own rock fraction through
# `Dx`/`Dy` — hence the reciprocal factor here. Without it the pressure of a cut cell
# relaxes `ϕ.center` times slower than the rest of the domain and is left visibly
# under-converged along a free surface. Every consumer multiplies at least one rock
# fraction back in (`γ_eff * RP` and the single center-fraction factor used by the
# Gershgorin bound), so no second volume-fraction factor is introduced; `isvalid_c`
# guarantees the divisor is positive.
@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, η, ϕ::JustRelax.RockRatio, η_mean, γfact, dt)

    if isvalid_c(ϕ, I...)
        ratios = @cell phase_ratios_center[I...]
        Kbdt = fn_ratio(get_bulk_modulus, rheology, ratios) * dt
        ηb[I...] = Kbdt

        η_local = η[I...]
        γ_num = γfact * (isinf(η_local) ? η_mean : η_local)
        γ_phy = isinf(Kbdt) ? γ_num : Kbdt
        γ_eff[I...] = γ_phy * γ_num / (γ_phy + γ_num) / ϕ.center[I...]
    else
        ηb[I...] = 0.0e0
        γ_eff[I...] = 0.0e0
    end

    return nothing
end
