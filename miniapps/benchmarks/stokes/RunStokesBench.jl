import Pkg; Pkg.activate(".")
using ParallelStencil
using JustRelax
# using GeoParams
using Printf, LinearAlgebra, GLMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# chose benchmark
# available options = :solvi, :solcx, :solkz
benchmark = :solkz 

# model resolution (number of gridpoints)
nx, ny = 512, 512 

# :single for a single run model with nx, ny resolution
# :multiple for grid sensitivy error plot
runtype = :multiple

if benchmark == :solcx
    # benchmark reference:
    # Duretz, Thibault, et al. "Discretization errors and free surface stabilization
    # in the finite difference and marker‐in‐cell method for applied geodynamics: 
    # A numerical study." Geochemistry, Geophysics, Geosystems 12.7 (2011).
    # DOI: 10.1029/2011GC003567
    # include plotting and error related functions
    include("solcx/SolCx.jl")
    
    # viscosity contrast
    Δη = 1e6

    if runtype == :single
        # run model
        geometry, stokes, ρ = solCx(Δη, nx=nx, ny=ny);
            
        # plot model output and error
        f = plot_solCx_error(geometry, stokes, Δη)
        
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
        geometry, stokes, ρ = solKz(nx=nx, ny=ny);
    
        # plot model output and error
        f = plot_solKz_error(geometry, stokes)
    
    elseif runtype == :multiple
        f = multiple_solKz(N = 10) # nx = ny = 2^(4:N)-1

    end
end