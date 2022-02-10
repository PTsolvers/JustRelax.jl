import Pkg; Pkg.activate(".")
using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
using JustRelax
using Printf, LinearAlgebra, GLMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# This main function for "Application" code is divided into 7 stages which
# are intended to cover usage with the GPU4GEO project and potential users
# of the software developed within it.

# 1. Quantities needed to describe "where the problem lives", in terms of (parallel) topology
# 2. Initialize tools which can represent this domain concretely in parallel (IGG here, could be PETSc/DM)
# 3. Concrete representations of data and population of values
#    - Includes information on embedding/coordinates
# 4. Tools, dependent on the data representation, to actually solve a particular physical problem (here JustRelax.jl, but could be PETSc's SNES)
#    - Note that here, the physical timestepping scheme is baked into this "physical problem"
# 5. Analysis and output which depends on the details of the solver
# 6. "Application" Analysis and output which does not depend on the details of the solver
# 7. Finalization/Cleanup
   
# include benchmark related functions
include("SolVi.jl")

function solvi()
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (256-1, 256-1) # number of nodes in x- and y-
    li = (10.0, 10.0)  # domain length in x- and y-
    geometry = Geometry(ni, li) # structure containing topology information

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(geometry)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(geometry)

    ## Setup-specific parameters and fields
    η0 = 1.0  # matrix viscosity
    ηi = 1e-3 # inclusion viscosity
    εbg = 1.0 # background strain-rate
    rc = 1.0  # clast radius
    η = solvi_viscosity(geometry, η0, ηi) # viscosity field

    ## Boundary conditions
    pureshear_bc!(stokes, geometry, εbg) 
    freeslip = (
        freeslip_x = true,
        freeslip_y = true
    )

    # Physical time loop
    t = 0.0
    ρ = @ones(size(η))
    while t < ttot
        solve!(stokes, pt_stokes, geometry, freeslip, ρ, η; iterMax = 10e3)
        t += Δt
    end

    ## PLOTS - Compare pressure against analytical solution
    Psolvi, = solvi_solution(geometry, η0, ηi, εbg, rc)
end


