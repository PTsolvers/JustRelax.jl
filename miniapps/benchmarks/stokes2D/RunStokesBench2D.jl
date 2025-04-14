using JustRelax, JustRelax.JustRelax2D, Printf, LinearAlgebra
using MPI: MPI
using GLMakie

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

# choose benchmark
benchmark = :solcx

# model resolution (number of gridpoints)
nx, ny = 64, 64

# set MPI
finalize_MPI = false

# :single for a single run model with nx, ny resolution
# :multiple for grid sensitivity error plot
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
    Δη = 1.0e6
    if runtype == :single
        # run model
        geometry, stokes, iters, ρ = solCx(
            Δη;
            nx = nx,
            ny = ny,
            init_MPI = MPI.Initialized() ? false : true,
            finalize_MPI = finalize_MPI,
        )

        # plot model output and error
        f = plot_solCx_error(geometry, stokes, Δη; cmap = :romaO)

    elseif runtype == :multiple
        f = multiple_solCx(; Δη = Δη, nrange = 6:10) # nx = ny = 2^(nrange)-1
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
    Δη = 1.0e6
    if runtype == :single
        # run model
        geometry, stokes, iters, = solKz(;
            Δη = Δη,
            nx = nx,
            ny = ny,
            init_MPI = MPI.Initialized() ? false : true,
            finalize_MPI = finalize_MPI,
        )

        # plot model output and error
        f = plot_solKz_error(geometry, stokes; cmap = :romaO)

    elseif runtype == :multiple
        f = multiple_solKz(; Δη = Δη, nrange = 4:10) # nx = ny = 2^(nrange)-1
    end

elseif benchmark == :solvi
    # Benchmark reference:
    #   D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
    #   general shear. Geophysical Journal International, 155(1):269–288, 2003.

    # include plotting and error related functions
    include("solvi/SolVi.jl") # need to call this again if we switch from gpu <-/-> cpu

    # model specific parameters
    Δη = 1.0e-3 # viscosity ratio between matrix and inclusion
    rc = 0.2 # radius of the inclusion
    εbg = 1.0e0 # background strain rate
    lx, ly = 2.0e0, 2.0e0 # domain siye in x and y directions
    if runtype == :single
        # run model
        geometry, stokes, iters = solVi(;
            Δη = Δη,
            nx = nx,
            ny = ny,
            lx = lx,
            ly = ly,
            rc = rc,
            εbg = εbg,
            init_MPI = MPI.Initialized() ? false : true,
            finalize_MPI = finalize_MPI,
        )

        # plot model output and error
        f = plot_solVi_error(geometry, stokes, Δη, εbg, rc)

    elseif runtype == :multiple
        f = multiple_solVi(; Δη = Δη, lx = lx, ly = ly, rc = rc, εbg = εbg, nrange = 4:8) # nx = ny = 2^(nrange)-1
    end

elseif benchmark == :solviel

    # include plotting and error related functions
    include("solvi/SolViEl.jl") # need to call this again if we switch from gpu <-/-> cpu

    # model specific parameters
    Δη = 1.0e1 # viscosity ratio between matrix and inclusion
    rc = 0.2 # radius of the inclusion
    εbg = 1.0e0 # background strain rate
    lx, ly = 2.0e0, 2.0e0 # domain siye in x and y directions
    if runtype == :single
        # run model
        geometry, stokes, iters = solViEl(;
            Δη = Δη,
            nx = nx,
            ny = ny,
            lx = lx,
            ly = ly,
            rc = rc,
            εbg = εbg,
            init_MPI = MPI.Initialized() ? false : true,
            finalize_MPI = finalize_MPI,
        )

        # plot model output and error
        f = plot_solVi_error(geometry, stokes, Δη, εbg, rc)

    elseif runtype == :multiple
        f = multiple_solViEl(; Δη = Δη, lx = lx, ly = ly, rc = rc, εbg = εbg, nrange = 4:8) # nx = ny = 2^(nrange)-1
    end

elseif benchmark == :elastic_buildup
    # Benchmark reference:
    #   Gerya, T. V., & Yuen, D. A. (2007). Robust characteristics method for
    #   modelling multiphase visco-elasto-plastic thermo-mechanical problems.
    #   Physics of the Earth and Planetary Interiors, 163(1-4), 83-105.

    # include plotting and error related functions
    include("elastic_buildup/Elastic_BuildUp.jl") # need to call this again if we switch from gpu <-/-> cpu

    # model specific parameters
    endtime = 125 # duration of the model in kyrs
    η0 = 1.0e22 # viscosity
    εbg = 1.0e-14 # background strain rate (pure shear boundary conditions)
    G = 10.0e9 # shear modulus
    lx, ly = 100.0e3, 100.0e3 # length of the domain in meters
    if runtype == :single
        # run model
        geometry, stokes, av_τyy, sol_τyy, t, iters = elastic_buildup(;
            nx = nx,
            ny = ny,
            lx = lx,
            ly = ly,
            endtime = endtime,
            η0 = η0,
            εbg = εbg,
            G = G,
            init_MPI = MPI.Initialized() ? false : true,
            finalize_MPI = finalize_MPI,
        )
        # plot model output and error
        f = plot_elastic_buildup(av_τyy, sol_τyy, t)

    elseif runtype == :multiple
        f = multiple_elastic_buildup(;
            lx = lx, ly = ly, endtime = endtime, η0 = η0, εbg = εbg, G = G, nrange = 4:8
        )
    end

else
    throw("Benchmark not available.")
end
