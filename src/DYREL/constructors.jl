function DYREL(ni::NTuple{2}; ϵ=1e-6, CFL= 0.99, c_fat = 0.5)
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
    JustRelax.DYREL{T, F}(γ_eff, Dx, Dy, λmaxVx, λmaxVy, dVxdτ, dVydτ, dτVx, dτVy,
                     dVx, dVy, βVx, βVy, cVx, cVy, αVx, αVy, ηb, CFL, ϵ, c_fat)
end

DYREL(nx::Integer, ny::Integer; ϵ=1e-6, CFL= 0.99, c_fat = 0.5) = DYREL((nx, ny); ϵ=ϵ, CFL=CFL, c_fat=c_fat)

function DYREL(::Type{CPUBackend}, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ=1e-6, CFL= 0.99, c_fat = 0.5, γfact = 20.0)
    return DYREL(stokes, rheology, phase_ratios, di, dt; ϵ=ϵ, CFL=CFL, c_fat=c_fat, γfact=γfact)
end

function DYREL(stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; ϵ=1e-6, CFL= 0.99, c_fat = 0.5, γfact = 20.0)
    
    ni = size(stokes.P)
    
    # instantiate DYREL object
    dyrel = DYREL(ni; ϵ = ϵ, CFL = CFL, c_fat = c_fat)
    
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)
    
    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)
    
    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)

    return dyrel
end

function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, di, dt; CFL= 0.99,  γfact = 20.0)    
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)
    
    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)
    
    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)
    
    return nothing
end

# variational version
function DYREL!(dyrel::JustRelax.DYREL, stokes::JustRelax.StokesArrays, rheology, phase_ratios, ϕ, di, dt; CFL= 0.99,  γfact = 20.0)    
    # compute bulk viscosity and penalty parameter
    compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, ϕ, γfact, dt)
    
    # compute Gershgorin estimates for maximum eigenvalues and diagonal preconditioners
    Gershgorin_Stokes2D_SchurComplement!(dyrel.Dx, dyrel.Dy, dyrel.λmaxVx, dyrel.λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, di, dt)
    
    # compute damping coefficients
    update_dτV_α_β!(dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy, dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy, CFL)
    
    return nothing
end

function compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)
    @parallel compute_bulk_viscosity_and_penalty!(dyrel.ηb, dyrel.γ_eff, rheology, phase_ratios.center, mean(stokes.viscosity.η[.!isinf.(stokes.viscosity.η)]), γfact, dt)
    return nothing
end

@parallel_indices (I...) function compute_bulk_viscosity_and_penalty!(ηb, γ_eff, rheology, phase_ratios_center, η_mean, γfact, dt)

    # bulk viscosity
    ratios      = @cell phase_ratios_center[I...]
    Kb          = fn_ratio(get_bulk_modulus, rheology, ratios)
    Kb          = isinf(Kb) ? η_mean : Kb
    ηb[I...]    = Kb * dt

    # penalty parameter factor
    γ_num       = γfact * η_mean
    γ_phy       = Kb * dt
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
        ratios      = @cell phase_ratios_center[I...]
        Kb          = fn_ratio(get_bulk_modulus, rheology, ratios)
        Kb          = isinf(Kb) ? η_mean : Kb
        ηb[I...]    = Kb * dt * ϕ.center[I...]
        
        # penalty parameter factor
        γ_num       = γfact * η_mean
        γ_phy       = Kb * dt
        γ_eff[I...] = γ_phy * γ_num / (γ_phy + γ_num) * ϕ.center[I...]
    else
        ηb[I...]    = 0e0
        γ_eff[I...] = 0e0
    end

    return nothing
end