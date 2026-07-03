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
    ∂τc_∂ε = zero_field_tuple(Val(9), nx, ny)
    ∂τv_∂ε = zero_field_tuple(Val(9), nx + 1, ny + 1)
    ∂ΔPψc_∂ε = zero_field_tuple(Val(3), nx, ny)
    ∂ΔPψc_∂η = zero_field_tuple(Val(3), nx, ny)
    ∂τc_∂η = zero_field_tuple(Val(3), nx, ny)
    ∂τv_∂η = zero_field_tuple(Val(3), nx + 1, ny + 1)
    ∂ηc_∂ε = zero_field_tuple(Val(3), nx, ny)
    ∂ηv_∂ε = zero_field_tuple(Val(3), nx + 1, ny + 1)
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
    ∂Rx_∂ΔPψ = zero_field_tuple(Val(2), nx - 1, ny)
    ∂Ry_∂τyy = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂τxy = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂P = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂P_num = zero_field_tuple(Val(2), nx, ny - 1)
    ∂Ry_∂ΔPψ = zero_field_tuple(Val(2), nx, ny - 1)

    T = typeof(γ_eff)
    F = typeof(CFL)
    S = typeof(∂τc_∂ε)
    D = typeof(∂ΔPψc_∂ε)
    E = typeof(∂εxx_∂Vx)
    return JustRelax.DYREL{T, F, S, D, E}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, CFL, ϵ, ϵ_vel, c_fact,
        ∂τc_∂ε, ∂τv_∂ε, ∂ΔPψc_∂ε, ∂ΔPψc_∂η, ∂τc_∂η, ∂τv_∂η, ∂ηc_∂ε, ∂ηv_∂ε,
        ∂εxx_∂Vx, ∂εyy_∂Vx, ∂∇V_∂Vx, ∂εxx_∂Vy, ∂εyy_∂Vy, ∂∇V_∂Vy, ∂εxy_∂Vx, ∂εxy_∂Vy,
        ∂Rx_∂τxx, ∂Rx_∂τxy, ∂Rx_∂P, ∂Rx_∂P_num, ∂Rx_∂ΔPψ,
        ∂Ry_∂τyy, ∂Ry_∂τxy, ∂Ry_∂P, ∂Ry_∂P_num, ∂Ry_∂ΔPψ
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
    ∂τc_∂ε = zero_field_tuple(Val(1), 1, 1, 1)
    ∂τv_∂ε = zero_field_tuple(Val(1), 1, 1, 1)
    ∂ΔPψc_∂ε = zero_field_tuple(Val(1), 1, 1, 1)
    ∂ΔPψc_∂η = zero_field_tuple(Val(1), 1, 1, 1)
    ∂τc_∂η = zero_field_tuple(Val(1), 1, 1, 1)
    ∂τv_∂η = zero_field_tuple(Val(1), 1, 1, 1)
    ∂ηc_∂ε = zero_field_tuple(Val(1), 1, 1, 1)
    ∂ηv_∂ε = zero_field_tuple(Val(1), 1, 1, 1)
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
    ∂Rx_∂ΔPψ = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂τyy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂τxy = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂P = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂P_num = zero_field_tuple(Val(1), 1, 1, 1)
    ∂Ry_∂ΔPψ = zero_field_tuple(Val(1), 1, 1, 1)

    T = typeof(γ_eff)
    F = typeof(CFL)
    S = typeof(∂τc_∂ε)
    D = typeof(∂ΔPψc_∂ε)
    E = typeof(∂εxx_∂Vx)
    return JustRelax.DYREL{T, F, S, D, E}(
        γ_eff, Dx, Dy, Dz, λmaxVx, λmaxVy, λmaxVz, dVxdτ, dVydτ, dVzdτ, dτVx, dτVy, dτVz,
        dVx, dVy, dVz, βVx, βVy, βVz, cVx, cVy, cVz, αVx, αVy, αVz, ηb, CFL, ϵ, ϵ_vel, c_fact,
        ∂τc_∂ε, ∂τv_∂ε, ∂ΔPψc_∂ε, ∂ΔPψc_∂η, ∂τc_∂η, ∂τv_∂η, ∂ηc_∂ε, ∂ηv_∂ε,
        ∂εxx_∂Vx, ∂εyy_∂Vx, ∂∇V_∂Vx, ∂εxx_∂Vy, ∂εyy_∂Vy, ∂∇V_∂Vy, ∂εxy_∂Vx, ∂εxy_∂Vy,
        ∂Rx_∂τxx, ∂Rx_∂τxy, ∂Rx_∂P, ∂Rx_∂P_num, ∂Rx_∂ΔPψ,
        ∂Ry_∂τyy, ∂Ry_∂τxy, ∂Ry_∂P, ∂Ry_∂P_num, ∂Ry_∂ΔPψ
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

    # instantiate DYREL object
    dyrel = DYREL(ni; ϵ = ϵ, ϵ_vel = ϵ_vel, CFL = CFL, c_fact = c_fact)

    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

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
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

    return nothing
end

function DYREL_AD!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, grid::Geometry, dt; CFL = 0.99, γfact = 20.0)
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

    # assemble Gershgorin estimates from local stress gradients
    Gershgorin_Stokes2D_SchurComplementAD(dyrel, grid._di.center, grid._di.vertex, grid._di.velocity[1], grid._di.velocity[2])

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

    return nothing
end

# variational version
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ, di, dt; CFL = 0.99, γfact = 20.0)
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)

    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)

    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

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
