"""
    struct DYREL{T, F}

Structure containing parameters and arrays for the DYREL (Dynamic Relaxation) solver.

# Fields
- `γ_eff`: Effective penalty parameter.
- `Dx`, `Dy`, `Dz`: Diagonal preconditioners for velocity updates in x, y, (and z) directions.
- `λmaxVx`, `λmaxVy`, `λmaxVz`: Maximum eigenvalues for stability calculation.
- `dVxdτ`, `dVydτ`, `dVzdτ`: Pseudo-time step related damping terms.
- `dτVx`, `dτVy`, `dτVz`: Pseudo-time steps for velocity fields.
- `dVx`, `dVy`, `dVz`: Velocity increments for the current iteration.
- `βVx`, `βVy`, `βVz`: Damping coefficients for momentum equation.
- `cVx`, `cVy`, `cVz`: Damping coefficients related to dynamic relaxation.
- `αVx`, `αVy`, `αVz`: Scaling factors for damping.
- `ηb`: Bulk viscosity field.
- `P_num`: Numerical pressure scratch field.
- `Rx0`, `Ry0`, `Rz0`: Velocity residual history scratch fields.
- `CFL`: Courant-Friedrichs-Lewy number.
- `ϵ`: General convergence tolerance.
- `ϵ_vel`: Velocity convergence tolerance.
- `c_fact`: Damping scaling factor.
"""
struct DYREL{T, F}
    γ_eff::T  # penalty parameter
    Dx::T     # diagonal preconditioner
    Dy::T     # diagonal preconditioner
    Dz::T     # diagonal preconditioner (3D)
    λmaxVx::T # maximum eigenvalue in x-direction
    λmaxVy::T # maximum eigenvalue in y-direction
    λmaxVz::T # maximum eigenvalue in z-direction (3D)
    dVxdτ::T  # damping coefficients
    dVydτ::T  # damping coefficients
    dVzdτ::T  # damping coefficients (3D)
    dτVx::T   # damping coefficients
    dτVy::T   # damping coefficients
    dτVz::T   # damping coefficients (3D)
    dVx::T    # damping coefficients
    dVy::T    # damping coefficients
    dVz::T    # damping coefficients (3D)
    βVx::T    # damping coefficients
    βVy::T    # damping coefficients
    βVz::T    # damping coefficients (3D)
    cVx::T    # damping coefficients
    cVy::T    # damping coefficients
    cVz::T    # damping coefficients (3D)
    αVx::T    # damping coefficients
    αVy::T    # damping coefficients
    αVz::T    # damping coefficients (3D)
    ηb::T     # bulk viscosity
    P_num::T  # numerical pressure scratch
    Rx0::T    # velocity residual history scratch
    Ry0::T    # velocity residual history scratch
    Rz0::T    # velocity residual history scratch (3D)
    CFL::F    # Courant-Friedrichs-Lewy condition
    ϵ::F      # convergence criterion
    ϵ_vel::F  # convergence criterion
    c_fact::F # damping factor
    ∂τxxc_∂εxx::T
    ∂τyyc_∂εyy::T
    ∂τxyv_∂εxy::T
end
