# DYREL struct supporting both 2D and 3D
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
    CFL::F    # Courant-Friedrichs-Lewy condition
    ϵ::F      # convergence criterion
    ϵ_vel::F  # convergence criterion
    c_fact::F # damping factor
end

