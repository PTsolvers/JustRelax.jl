"""
    DYREL(ni::NTuple{N, Integer}; ϵ=1e-6, ϵ_vel=1e-6, CFL=0.99, c_fact=0.5) where N

Creates a new `DYREL` struct with fields initialized to zero.

# Arguments
- `ni`: Tuple containing the grid dimensions `(nx, ny)` for 2D or `(nx, ny, nz)` for 3D.
- `ϵ`: General convergence tolerance.
- `ϵ_vel`: Velocity convergence tolerance.
- `CFL`: Courant-Friedrichs-Lewy number.
- `c_fact`: Damping scaling factor.
"""
@inline zero_field_tuple(::Val{N}, dims...) where {N} =
    ntuple(_ -> @zeros(dims...), Val(N))

function DYREL(ni::NTuple{2}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5)
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
    ∂τxxc_∂εxx = @zeros(nx, ny)
    ∂τxxc_∂εyy = @zeros(nx, ny)
    ∂τxxc_∂εxy = @zeros(nx, ny)
    ∂τyyc_∂εxx = @zeros(nx, ny)
    ∂τyyc_∂εyy = @zeros(nx, ny)
    ∂τyyc_∂εxy = @zeros(nx, ny)
    ∂τxyc_∂εxx = @zeros(nx, ny)
    ∂τxyc_∂εyy = @zeros(nx, ny)
    ∂τxyc_∂εxy = @zeros(nx, ny)
    ∂τxxv_∂εxx = @zeros(nx + 1, ny + 1)
    ∂τxxv_∂εyy = @zeros(nx + 1, ny + 1)
    ∂τxxv_∂εxy = @zeros(nx + 1, ny + 1)
    ∂τyyv_∂εxx = @zeros(nx + 1, ny + 1)
    ∂τyyv_∂εyy = @zeros(nx + 1, ny + 1)
    ∂τyyv_∂εxy = @zeros(nx + 1, ny + 1)
    ∂τxyv_∂εxx = @zeros(nx + 1, ny + 1)
    ∂τxyv_∂εyy = @zeros(nx + 1, ny + 1)
    ∂τxyv_∂εxy = @zeros(nx + 1, ny + 1)
    ∂εxx_∂Vx = zero_field_tuple(Val(2), nx, ny)
    ∂εyy_∂Vx = zero_field_tuple(Val(2), nx, ny)
    ∂∇V_∂Vx = zero_field_tuple(Val(2), nx, ny)
    ∂εxx_∂Vy = zero_field_tuple(Val(2), nx, ny)
    ∂εyy_∂Vy = zero_field_tuple(Val(2), nx, ny)
    ∂∇V_∂Vy = zero_field_tuple(Val(2), nx, ny)
    ∂εxy_∂Vx = zero_field_tuple(Val(2), nx + 1, ny + 1)
    ∂εxy_∂Vy = zero_field_tuple(Val(2), nx + 1, ny + 1)
    ∂Rx_∂τxx = zero_field_tuple(Val(2), nx - 1, ny)
    ∂Rx_∂τxy = zero_field_tuple(Val(2), nx - 1, ny)
    ∂Rx_∂P = zero_field_tuple(Val(2), nx - 1, ny)
    ∂Rx_∂P_num = zero_field_tuple(Val(2), nx - 1, ny)
    ∂Ry_∂τyy = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂τxy = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂P = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂P_num = zero_field_tuple(Val(2), nx, ny - 1)
    P_num = @zeros(nx, ny)
    Rx0 = @zeros(nx - 1, ny)
    Ry0 = @zeros(nx, ny - 1)
    Rz0 = @zeros(1, 1)  # dummy for 2D

    T = typeof(γ_eff)
    F = typeof(CFL)
    E = typeof(∂εxx_∂Vx)
    return JustRelax.DYREL{T, F, E}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0, Rz0,
        CFL, ϵ, ϵ_vel, c_fact,
        ∂τxxc_∂εxx, ∂τxxc_∂εyy, ∂τxxc_∂εxy, ∂τyyc_∂εxx, ∂τyyc_∂εyy, ∂τyyc_∂εxy, ∂τxyc_∂εxx, ∂τxyc_∂εyy, ∂τxyc_∂εxy,
        ∂τxxv_∂εxx, ∂τxxv_∂εyy, ∂τxxv_∂εxy, ∂τyyv_∂εxx, ∂τyyv_∂εyy, ∂τyyv_∂εxy, ∂τxyv_∂εxx, ∂τxyv_∂εyy, ∂τxyv_∂εxy,
        ∂εxx_∂Vx, ∂εyy_∂Vx, ∂∇V_∂Vx, ∂εxx_∂Vy, ∂εyy_∂Vy, ∂∇V_∂Vy, ∂εxy_∂Vx, ∂εxy_∂Vy, ∂Rx_∂τxx, ∂Rx_∂τxy, ∂Rx_∂P,
        ∂Rx_∂P_num, ∂Ry_∂τyy, ∂Ry_∂τxy, ∂Ry_∂P, ∂Ry_∂P_num
    )
end

DYREL(nx::Integer, ny::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5) = DYREL((nx, ny); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)

function DYREL(ni::NTuple{3}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5)
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
    ∂τxxc_∂εxx = @zeros(1, 1, 1)
    ∂τxxc_∂εyy = @zeros(1, 1, 1)
    ∂τxxc_∂εxy = @zeros(1, 1, 1)
    ∂τyyc_∂εxx = @zeros(1, 1, 1)
    ∂τyyc_∂εyy = @zeros(1, 1, 1)
    ∂τyyc_∂εxy = @zeros(1, 1, 1)
    ∂τxyc_∂εxx = @zeros(1, 1, 1)
    ∂τxyc_∂εyy = @zeros(1, 1, 1)
    ∂τxyc_∂εxy = @zeros(1, 1, 1)
    ∂τxxv_∂εxx = @zeros(1, 1, 1)
    ∂τxxv_∂εyy = @zeros(1, 1, 1)
    ∂τxxv_∂εxy = @zeros(1, 1, 1)
    ∂τyyv_∂εxx = @zeros(1, 1, 1)
    ∂τyyv_∂εyy = @zeros(1, 1, 1)
    ∂τyyv_∂εxy = @zeros(1, 1, 1)
    ∂τxyv_∂εxx = @zeros(1, 1, 1)
    ∂τxyv_∂εyy = @zeros(1, 1, 1)
    ∂τxyv_∂εxy = @zeros(1, 1, 1)
    ∂εxx_∂Vx = zero_field_tuple(Val(1), 1, 1, 1)
    ∂εyy_∂Vx = zero_field_tuple(Val(1), 1, 1, 1)
    ∂∇V_∂Vx = zero_field_tuple(Val(1), 1, 1, 1)
    ∂εxx_∂Vy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂εyy_∂Vy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂∇V_∂Vy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂εxy_∂Vx = zero_field_tuple(Val(1), 1, 1, 1)
    ∂εxy_∂Vy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Rx_∂τxx = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Rx_∂τxy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Rx_∂P = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Rx_∂P_num = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂τyy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂τxy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂P = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂P_num = zero_field_tuple(Val(1), 1, 1, 1)
    P_num = @zeros(nx, ny, nz)
    Rx0 = @zeros(nx - 1, ny, nz)
    Ry0 = @zeros(nx, ny - 1, nz)
    Rz0 = @zeros(nx, ny, nz - 1)

    T = typeof(γ_eff)
    F = typeof(CFL)
    E = typeof(∂εxx_∂Vx)
    return JustRelax.DYREL{T, F, E}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, P_num, Rx0, Ry0, Rz0,
        CFL, ϵ, ϵ_vel, c_fact,
        ∂τxxc_∂εxx, ∂τxxc_∂εyy, ∂τxxc_∂εxy, ∂τyyc_∂εxx, ∂τyyc_∂εyy, ∂τyyc_∂εxy, ∂τxyc_∂εxx, ∂τxyc_∂εyy, ∂τxyc_∂εxy,
        ∂τxxv_∂εxx, ∂τxxv_∂εyy, ∂τxxv_∂εxy, ∂τyyv_∂εxx, ∂τyyv_∂εyy, ∂τyyv_∂εxy, ∂τxyv_∂εxx, ∂τxyv_∂εyy, ∂τxyv_∂εxy,
        ∂εxx_∂Vx, ∂εyy_∂Vx, ∂∇V_∂Vx, ∂εxx_∂Vy, ∂εyy_∂Vy, ∂∇V_∂Vy, ∂εxy_∂Vx, ∂εxy_∂Vy,
        ∂Rx_∂τxx, ∂Rx_∂τxy, ∂Rx_∂P, ∂Rx_∂P_num, ∂Ry_∂τyy, ∂Ry_∂τxy, ∂Ry_∂P, ∂Ry_∂P_num
    )
end

DYREL(nx::Integer, ny::Integer, nz::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5) = DYREL((nx, ny, nz); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)


DYREL(::Type{CPUBackend}, ni::NTuple{N, Integer}; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5) where {N} = DYREL(ni; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)
DYREL(::Type{CPUBackend}, nx::Integer, ny::Integer, nz::Integer; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5) = DYREL((nx, ny, nz); ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)

function DYREL(::Type{CPUBackend}, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ = 1.0e-6, ϵ_vel = 1.0e-6, CFL = 0.99, c_fact = 0.5, γfact = 20.0)
    return DYREL(stokes, rheology, phase_ratios, di, dt; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact, γfact = γfact)
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

    ni = size(stokes.P)
    dim = Val(length(ni))

    # instantiate DYREL object
    dyrel = DYREL(ni; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes_SchurComplement!(dim, dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return dyrel
end


"""
    DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; CFL=0.99, γfact=20.0)

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
- `CFL`: Courant number (default: 0.99).
- `γfact`: Penalty factor (default: 20.0).

Returns `nothing`.
"""
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; CFL = 0.99, γfact = 20.0)
    dim = Val(ndims(stokes.P))

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes_SchurComplement!(dim, dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return nothing
end

function DYREL_AD!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, grid::Geometry, dt; CFL = 0.99, γfact = 20.0)
    dim = Val(ndims(stokes.P))

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # assemble Gershgorin estimates from local stress gradients
    Gershgorin_Stokes_SchurComplementAD!(dim, dyrel, grid)

    # compute damping coefficients
    update_dτV_α_β!(dyrel, CFL)

    return nothing
end

# variational version
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ, di, dt; CFL = 0.99, γfact = 20.0)
    dim = Val(ndims(stokes.P))

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes_SchurComplement!(dim, dyrel.Dx, dyrel.Dy, dyrel.Dz, dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

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
   - `γ_num = γfact * η_mean`
   - `γ_phy = Kb` (or related term)
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

    # penalty parameter factor
    γ_num = γfact * η_mean
    γ_phy = isinf(Kbdt) ? γfact * η_mean : Kbdt
    γ_eff[I...] = γ_phy * γ_num / (γ_phy + γ_num)

    return nothing
end


# variational version

function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)
    ni = size(stokes.P)
    @parallel (@idx ni) compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, ϕ, mean(stokes.viscosity.η[.!isinf.(stokes.viscosity.η)]), γfact, dt)
    return nothing
end

@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, ϕ, η_mean, γfact, dt)

    if isvalid_c(ϕ, I...)
        # bulk viscosity
        ratios = @cell phase_ratios_center[I...]
        Kb = fn_ratio(get_bulk_modulus, rheology, ratios)
        Kb = isinf(Kb) ? η_mean : Kb
        ηb[I...] = Kb * dt * ϕ.center[I...]

        # penalty parameter factor
        γ_num = γfact * η_mean
        γ_phy = Kb * dt
        γ_eff[I...] = γ_phy * γ_num / (γ_phy + γ_num) * ϕ.center[I...]
    else
        ηb[I...] = 0.0e0
        γ_eff[I...] = 0.0e0
    end

    return nothing
end
