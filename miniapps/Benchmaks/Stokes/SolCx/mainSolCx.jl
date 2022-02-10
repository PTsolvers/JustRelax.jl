import Pkg; Pkg.activate(".")
using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
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
include("SolCx.jl")

innfunloc(A, i, j, f)  = f(
        A[i,   j],
        A[i-1, j],
        A[i+1, j],
        A[i, j-1],
        A[i, j+1],
    )

function compute_funloc!(A, B, f::Function)

    Threads.@threads for i in 2:size(A,1)-1
        for j in 2:size(A,2)-1
            @inbounds  A[i, j] = innfunloc(A, i, j, f)
        end 
    end

    Threads.@threads for i in 2:size(A, 2)-1
        @inbounds B[i, 1] = f(
            A[i,     1],
            A[i+1,   1],
            A[i-1,   1],
            A[i,   1+1],
        )
        @inbounds B[i, size(A, 2)] = f(
            A[i,   size(A, 2)  ],
            A[i+1, size(A, 2)  ],
            A[i-1, size(A, 2)  ],
            A[i,   size(A, 2)-1],
        )
    end

end

mean(args...) = sum(args)/length(args)
harmmean(args...) = length(args)/sum(1.0./args)
geometricmean(args...) = reduce(*, args)^(1/length(args))

@parallel_indices (iy) function smooth_boundaries_x!(A::PTArray)
    A[iy, 1  ] = A[iy, 20    ]
    A[iy, 2  ] = A[iy, 20    ]
    A[iy, 3  ] = A[iy, 20    ]
    A[iy, 4  ] = A[iy, 20    ]
    A[iy, 5  ] = A[iy, 20    ]
    A[iy, 6  ] = A[iy, 20    ]
    A[iy, end] = A[iy, end-20]
    A[iy, end-1] = A[iy, end-20]
    A[iy, end-2] = A[iy, end-20]
    A[iy, end-3] = A[iy, end-20]
    A[iy, end-4] = A[iy, end-20]
    A[iy, end-5] = A[iy, end-20]
    return
end

function solCx(Δη; nx=256-1, ny=256-1, lx=1e0, ly=1e0)
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
    η = solCx_viscosity(geometry, Δη = Δη) # viscosity field
    ρ = solCx_density(geometry)
    fy = ρ*g

    # smooth viscosity jump (otherwise no convergence for Δη > ~15)
    η2 = deepcopy(η)
    for ism=1:30
        @parallel smooth!(η2, η, 1.0)
        η, η2 = η2, η
    end

    # we dont need to forget to smooth also the nodes at the borders 
    @parallel (1:size(η,1)) smooth_boundaries_x!(η)

    ## Boundary conditions
    freeslip = (
        freeslip_x = true,
        freeslip_y = true
    )

    # Physical time loop
    t = 0.0
    while t < ttot
        solve2!(stokes, pt_stokes, geometry, freeslip, fy, η; iterMax = 20e3, nout = 100)
        t += Δt
    end

    return geometry, stokes

end

function solve2!(stokes::StokesArrays, pt_stokes::PTStokesCoeffs, geometry::Geometry{2}, freeslip, ρg, η; iterMax = 10e3, nout = 500)
    # unpack
    dx, dy = geometry.di 
    lx, ly = geometry.li 
    (; Vx, Vy) = stokes.V
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    (; P, ∇V) = stokes
    (; Ry, Rx) = stokes.R
    (; Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ) = pt_stokes
    (;freeslip_x, freeslip_y) =freeslip

    # ~preconditioner
    ητ = deepcopy(η)
    # @parallel compute_maxloc!(ητ, η)
    compute_funloc!(ητ, η, geometricmean)
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, geometry.max_li)
    # errors
    err=2*ϵ; iter=0; err_evo1=Float64[]; err_evo2=Float64[]; err_rms = Float64[]
    
    # solver loop
    # Gdτ *= 1e-1
    # dτ_Rho *= 1e-3
    while err > ϵ && iter <= iterMax
        if (iter==11)  global wtime0 = Base.time()  end
        @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
        @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
        @parallel compute_dV!(Rx, Ry, dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
        @parallel compute_V!(Vx, Vy, dVx, dVy)

        # free slip boundary conditions
        if (freeslip_x) @parallel (1:size(Vx,1)) free_slip_y!(Vx) end
        if (freeslip_y) @parallel (1:size(Vy,2)) free_slip_x!(Vy) end

        iter += 1
        if (iter > 1) && (iter % nout == 0)
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            norm_Rx    = norm(Rx)/(Pmax-Pmin)*lx/sqrt(length(Rx))
            norm_Ry    = norm(Ry)/(Pmax-Pmin)*lx/sqrt(length(Ry))
            norm_∇V    = norm(∇V)/(Vmax-Vmin)*lx/sqrt(length(∇V))
            # norm_Rx = norm(Rx)/length(Rx); norm_Ry = norm(Ry)/length(Ry); norm_∇V = norm(∇V)/length(∇V)
            err = maximum([norm_Rx, norm_Ry, norm_∇V])
            push!(err_evo1, maximum([norm_Rx, norm_Ry, norm_∇V])); push!(err_evo2,iter)
            @printf("Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n", iter, err, norm_Rx, norm_Ry, norm_∇V)
        end
    end
end

geometry, stokes = solCx(1e6, nx=101*2, ny=101*2);
# plot model output
f1 = plot_solCx(geometry, stokes, cmap = :vik, fun = heatmap!)

# # # Compare pressure against analytical solution
# f2 = plot_solCx_error(geometry, stokes)

# function run_test(; N = 9)
#     N = 9
#     L2_vx, L2_vy, L2_p = zeros(N), zeros(N), zeros(N)
#     for i in 1:N
#         nx = ny = 32*i-1
#         geometry, stokes = solkz(nx=nx, ny=ny)
#         L2_vx[i], L2_vy[i], L2_p[i] = Li_error(geometry, stokes, order=2)
#     end

#     nx = @. 32*(1:N)-1
#     h = @. (1/nx)

#     f = Figure( fontsize=28) 
#     ax = Axis(f[1,1], yscale = log10, xscale = log10,  yminorticksvisible = true, yminorticks = IntervalsBetween(8))
#     lines!(ax, h, (L2_vx), linewidth=3, label = "Vx")
#     lines!(ax, h, (L2_vy), linewidth=3, label = "Vy")
#     lines!(ax, h, (L2_p),  linewidth=3, label = "P")
#     axislegend(ax)
#     ax.xlabel = "h"
#     ax.ylabel = "L2 norm"
#     display(f)

# end

# run_test(N =9)