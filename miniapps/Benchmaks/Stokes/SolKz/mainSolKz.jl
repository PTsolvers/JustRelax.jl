import Pkg; Pkg.activate(".")
using ParallelStencil
using JustRelax
using GeoParams
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
include("SolKz.jl")

function solkz(; nx=256-1, ny=256-1, lx=1e0, ly=1e0)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    geometry = Geometry(ni, li) # structure containing topology information
    g = 1

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(geometry)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(geometry)

    ## Setup-specific parameters and fields
    η = solKz_viscosity(geometry) # viscosity field
    ρ = solKz_density(geometry)
    fy = ρ*g

    ## Boundary conditions
    freeslip = (
        freeslip_x = true,
        freeslip_y = true
    )

    # Physical time loop
    t = 0.0
    while t < ttot
        solve!(stokes, pt_stokes, geometry, freeslip, fy, η; iterMax = 10e3)
        t += Δt
    end

    return geometry, stokes

end

# geometry, stokes = solkz(nx=31, ny=31)

# # plot model output
# f1 = plot_solkz(geometry, stokes)
# # Compare pressure against analytical solution
# f2 = plot_solkz_error(geometry, stokes)

function run_test(; N = 9)
    N = 9
    L2_vx, L2_vy, L2_p = zeros(N), zeros(N), zeros(N)
    for i in 1:N
        nx = ny = 32*i-1
        geometry, stokes = solkz(nx=nx, ny=ny)
        L2_vx[i], L2_vy[i], L2_p[i] = Li_error(geometry, stokes, order=2)
    end

    nx = @. 32*(1:N)-1
    h = @. (1/nx)

    f = Figure( fontsize=28) 
    ax = Axis(f[1,1], yscale = log10, xscale = log10,  yminorticksvisible = true, yminorticks = IntervalsBetween(8))
    lines!(ax, h, (L2_vx), linewidth=3, label = "Vx")
    lines!(ax, h, (L2_vy), linewidth=3, label = "Vy")
    lines!(ax, h, (L2_p),  linewidth=3, label = "P")
    axislegend(ax)
    ax.xlabel = "h"
    ax.ylabel = "L2 norm"
    display(f)

end

run_test(N =9)