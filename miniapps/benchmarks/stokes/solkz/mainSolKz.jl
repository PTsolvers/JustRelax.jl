using ParallelStencil
using JustRelax
using Printf, LinearAlgebra, GLMakie
using ParallelStencil.FiniteDifferences2D

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
# include("SolKz.jl")

function solKz_viscosity(xci, ni; B=log(1e6))
    xc, yc = xci
    # make grid array (will be eaten by GC)
    y = [yci for _ in xc, yci in yc]
    η = @zeros(ni...)
    # inner closure
    _viscosity(y, B) = exp(B*y)
    # outer closure
    @parallel function viscosity(η, y, B) 
        @all(η) =  _viscosity(@all(y), B)
        return
    end
    # compute viscosity
    @parallel viscosity(η, y, B) 

    return η
end

function solKz_density(xci, ni)
    xc, yc = xci
    # make grid array (will be eaten by GC)
    x = [xci for xci in xc, _ in yc]
    y = [yci for _ in xc, yci in yc]
    ρ = @zeros(ni...)
    # inner closure
    _density(x, y) = -sin(2*y)*cos(3*π*x)
    # outer closure
    @parallel function density(ρ, x, y) 
        @all(ρ) = _density(@all(x), @all(y))
        return
    end
    # compute viscosity
    @parallel density(ρ, x, y)

    return ρ
end

function solkz(; nx=256-1, ny=256-1, lx=1e0, ly=1e0)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li/ni # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells
    # geometry = Geometry(ni, li) # structure containing topology information
    g = 1 # gravity

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step
     
    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    η = solKz_viscosity(xci, ni) # viscosity field
    ρ = solKz_density(xci, ni)
    fy = ρ*g

    ## Boundary conditions
    freeslip = (
        freeslip_x = true,
        freeslip_y = true
    )

    # Physical time loop
    t = 0.0
    while t < ttot
        solve!(stokes, pt_stokes, di, li, max_li, freeslip, fy, η; iterMax = 10e3)
        t += Δt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes

end

geometry, stokes = solkz(nx=31, ny=31)

# # plot model output
f1 = plot_solkz(geometry, stokes)
# # Compare pressure against analytical solution
f2 = plot_solkz_error(geometry, stokes)

function run_test(; N = 10)
    
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]  
    for i in 4:N
        nx = ny = 2^i-1
        geometry, stokes = solkz(nx=nx, ny=ny)
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes, order=2)
        push!(L2_vxi, L2_vx[i])
        push!(L2_vyi, L2_vy[i])
        push!(L2_pi,  L2_p[i])
    end

    nx = @. 2^(4:N)-1
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

run_test(N=5)