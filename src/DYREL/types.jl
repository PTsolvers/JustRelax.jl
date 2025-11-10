 
# for now lets focus on the 2D DYREL model only 
struct DYREL{T, F}
    γ_eff::T  # penalty parameter
    Dx::T     # diagonal preconditioner
    Dy::T     # diagonal preconditioner
    λmaxVx::T # maximum eigenvalue in x-direction
    λmaxVy::T # maximum eigenvalue in y-direction
    dVxdτ::T  # damping coefficients
    dVydτ::T  # damping coefficients
    dτVx::T   # damping coefficients
    dτVy::T   # damping coefficients
    dVx::T    # damping coefficients
    dVy::T    # damping coefficients
    βVx::T    # damping coefficients
    βVy::T    # damping coefficients
    cVx::T    # damping coefficients
    cVy::T    # damping coefficients
    αVx::T    # damping coefficients
    αVy::T    # damping coefficients
    ηb::T     # bulk viscosity
    CFL::F    # Courant-Friedrichs-Lewy condition
    ϵ::F      # convergence criterion
    c_fact::F # damping factor
end
