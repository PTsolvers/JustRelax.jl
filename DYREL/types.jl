 
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
    c_fact::F # damping factor
end

DYREL(nx::Integer, ny::Integer; CFL= 0.99, c_fat = 0.5) = DYREL((nx, ny); CFL=CFL, c_fat=c_fat)

function DYREL(ni::NTuple{2}; CFL= 0.99, c_fat = 0.5)
    nx, ny = ni
    # penalty parameter
    γ_eff  = @zeros(nx, ny)
    # bulk viscosity
    ηb     = @zeros(nx, ny)
    # Diagonal preconditioner arrays
    Dx     = @zeros(nx-1, ny)
    Dy     = @zeros(nx, ny-1)
    # maximum eigenvalue estimates
    λmaxVx = @zeros(nx-1, ny)
    λmaxVy = @zeros(nx, ny-1)
    dVxdτ  = @zeros(nx-1, ny)
    dVydτ  = @zeros(nx, ny-1)
    dτVx   = @zeros(nx-1, ny)
    dτVy   = @zeros(nx, ny-1)
    dVx    = @zeros(nx-1, ny)
    dVy    = @zeros(nx, ny-1)
    βVx    = @zeros(nx-1, ny)
    βVy    = @zeros(nx, ny-1)
    cVx    = @zeros(nx-1, ny)
    cVy    = @zeros(nx, ny-1)
    αVx    = @zeros(nx-1, ny)
    αVy    = @zeros(nx, ny-1)
    
    T = typeof(γ_eff)
    F = typeof(CFL)
    DYREL{T, F}(γ_eff, Dx, Dy, λmaxVx, λmaxVy, dVxdτ, dVydτ, dτVx, dτVy, dVx, dVy, βVx, βVy, cVx, cVy, αVx, αVy, ηb, CFL, c_fat)
end
