using ParallelStencil
using Printf, LinearAlgebra, GLMakie
using JustRelax

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
   
# # include benchmark related functions
# include("SolVi.jl")

function solvi_viscosity(ni, di, li, rc, η0, ηi)
    dx, dy = di
    lx, ly = li
    Rad2 = [sqrt.(((ix-1)*dx +0.5*dx -0.5*lx)^2 + ((iy-1)*dy +0.5*dy -0.5*ly)^2) for ix=1:ni[1], iy=1:ni[2]]
    η    = fill(η0, ni...)
    η[Rad2.<rc] .= ηi
    η2 = deepcopy(η)
    for _ in 1:10
        @parallel smooth!(η2, η, 1.0)
        η, η2 = η2, η
    end
    η
end

function solvi(; nx=256-1, ny=256-1, lx=1e1, ly=1e1, rc = 1e0)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li/(ni-1) # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    η0 = 1.0  # matrix viscosity
    ηi = 1e-3 # inclusion viscosity
    εbg = 1.0 # background strain-rate
    rc = rc  # clast radius
    η = solvi_viscosity(ni, di, li, rc, η0, ηi) # viscosity field

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg) 
    freeslip = (
        freeslip_x = true,
        freeslip_y = true
    )

    # Physical time loop
    t = 0.0
    ρ = @zeros(size(stokes.P))
    while t < ttot
        solve!(stokes, pt_stokes, di, li, max_li, freeslip, ρ, η; iterMax = 10e3)
        t += Δt
    end

    return  (ni=ni, xci=xci, xvi=xvi, li=li), stokes

end

geometry, stokes = solvi()

## PLOTS - Compare pressure against analytical solution
# Psolvi, = solvi_solution(geometry, η0, ηi, εbg, rc)

