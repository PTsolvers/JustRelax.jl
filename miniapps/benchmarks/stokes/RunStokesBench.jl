using ParallelStencil
using JustRelax
using Printf, LinearAlgebra, GLMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# choose benchmark
# available options = :solvi, :solcx, :solkz
benchmark = :solcx

# model resolution (number of gridpoints)
nx, ny = 128, 128

# :single for a single run model with nx, ny resolution
# :multiple for grid sensitivy error plot
runtype = :single

if benchmark == :solcx
    # benchmark reference:
    # Duretz, Thibault, et al. "Discretization errors and free surface stabilization
    # in the finite difference and marker‐in‐cell method for applied geodynamics: 
    # A numerical study." Geochemistry, Geophysics, Geosystems 12.7 (2011).
    # DOI: 10.1029/2011GC003567
    
    # include plotting and error related functions
    include("solcx/SolCx.jl") # need to call this again if we switch from gpu <-/-> cpu
    
    # viscosity contrast
    Δη = 1e6

    if runtype == :single
        # run model
        geometry, stokes, iters, ρ = solCx(Δη, nx=nx, ny=ny);
            
        # plot model output and error
        f = plot_solCx_error(geometry, stokes, Δη, cmap=:romaO)
        
    elseif runtype == :multiple
        f = multiple_solCx(Δη = Δη, N = 10) # nx = ny = 2^(6:N)-1

    end

elseif benchmark == :solkz
    # benchmark reference:
    # Duretz, Thibault, et al. "Discretization errors and free surface stabilization
    # in the finite difference and marker‐in‐cell method for applied geodynamics: 
    # A numerical study." Geochemistry, Geophysics, Geosystems 12.7 (2011).
    # DOI: 10.1029/2011GC003567

    # include plotting and error related functions
    include("solkz/SolKz.jl")
        
    # viscosity contrast: Δη = 1e6
    if runtype == :single
        # run model
        geometry, stokes, iters, ρ = solKz(nx=nx, ny=ny);
    
        # plot model output and error
        f = plot_solKz_error(geometry, stokes, cmap=:romaO)
    
    elseif runtype == :multiple
        f = multiple_solKz(N = 10) # nx = ny = 2^(4:N)-1

    end
end
