using ParallelStencil
using JustRelax
using Printf, LinearAlgebra, CairoMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# choose benchmark
# available options = :solvi, :solcx, :solkz
benchmark = :solvi

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
        f = multiple_solCx(Δη = Δη, nrange = 6:10) # nx = ny = 2^(nrange)-1

    end

elseif benchmark == :solkz
    # benchmark reference:
    # Duretz, Thibault, et al. "Discretization errors and free surface stabilization
    # in the finite difference and marker‐in‐cell method for applied geodynamics: 
    # A numerical study." Geochemistry, Geophysics, Geosystems 12.7 (2011).
    # DOI: 10.1029/2011GC003567

    # include plotting and error related functions
    include("solkz/SolKz.jl")
        
    # viscosity contrast
    Δη = 1e6
    if runtype == :single
        # run model
        geometry, stokes, iters,  = solKz(Δη=Δη, nx=nx, ny=ny);
    
        # plot model output and error
        f = plot_solKz_error(geometry, stokes, cmap=:romaO)
    
    elseif runtype == :multiple
        f = multiple_solKz(Δη = Δη, nrange = 4:10) # nx = ny = 2^(nrange)-1

    end
    
elseif benchmark == :solvi
    # Benchmark reference:
    #   D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
    #   general shear. Geophysical Journal International, 155(1):269–288, 2003.
        
    # include plotting and error related functions
    include("solvi/SolVi.jl") # need to call this again if we switch from gpu <-/-> cpu
    
    # model specific parameters
    Δη = 1e-3 # viscosity ratio between matrix and inclusion
    rc = 0.2 # radius of the inclusion
    εbg = 1e0 # background strain rate
    lx, ly = 2e0, 2e0 # domain siye in x and y directions
    if runtype == :single
        # run model
        geometry, stokes, iters = solVi(Δη=Δη, nx=nx, ny=ny, lx=lx, ly=ly, rc = rc, εbg = εbg);
            
        # plot model output and error
        f = plot_solVi_error(geometry, stokes, Δη, εbg, rc)
        
    elseif runtype == :multiple
        f = multiple_solVi(; Δη=Δη, lx=lx, ly=ly, rc = rc, εbg = εbg, nrange=4:8) # nx = ny = 2^(nrange)-1
    
    end

elseif benchmark == :solviel
   
    # include plotting and error related functions
    include("solvi/SolViEl.jl") # need to call this again if we switch from gpu <-/-> cpu
    
    # model specific parameters
    Δη = 1e-3 # viscosity ratio between matrix and inclusion
    rc = 0.2 # radius of the inclusion
    εbg = 1e0 # background strain rate
    lx, ly = 2e0, 2e0 # domain siye in x and y directions
    if runtype == :single
        # run model
        geometry, stokes, iters = solViEl(Δη=Δη, nx=nx, ny=ny, lx=lx, ly=ly, rc = rc, εbg = εbg);
            
        # plot model output and error
        f = plot_solVi_error(geometry, stokes, Δη, εbg, rc)
        
    elseif runtype == :multiple
        f = multiple_solVi(; Δη=Δη, lx=lx, ly=ly, rc = rc, εbg = εbg, nrange=4:8) # nx = ny = 2^(nrange)-1
    
    end

elseif benchmark == :elastic_buildup
   
    # include plotting and error related functions
    include("solelastic/SolElastic.jl") # need to call this again if we switch from gpu <-/-> cpu

    # model specific parameters
    endtime = 500 # duration of the model in kyrs
    η0 = 1e22 # viscosity
    εbg = 1e-14 # background strain rate (pure shear boundary conditions)
    G = 10e9 # shear modulus
    lx, ly = 100e3, 100e3 # length of the domain in meters
    if runtype == :single
        # run model
        geometry, stokes, av_τyy, sol_τyy, t, iters = 
            elastic_buildup( nx=nx, ny=ny, lx=lx, ly=ly, endtime = endtime, η0 = η0, εbg = εbg, G = G)
        # plot model output and error
        f = plot_elasic_buildup(av_τyy, sol_τyy, t) 

    elseif runtype == :multiple
        # f = multiple_solVi(; Δη=Δη, lx=lx, ly=ly, rc = rc, εbg = εbg, nrange=4:8) # nx = ny = 2^(nrange)-1
    
    end

else
    throw("Benchmark not available.")
    
end
