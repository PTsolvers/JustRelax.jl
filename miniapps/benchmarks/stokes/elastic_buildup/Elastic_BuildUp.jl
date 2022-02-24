import Statistics: mean 

# Benchmark reference:
#   Gerya, T. V., & Yuen, D. A. (2007). Robust characteristics method for
#   modelling multiphase visco-elasto-plastic thermo-mechanical problems.
#   Physics of the Earth and Planetary Interiors, 163(1-4), 83-105.

# Analytical solution
solution(ε, t, G, η) = 2*ε*η*(1-exp(-G*t/η))

function plot_elasic_buildup(av_τyy, sol_τyy, t)
    f = Figure(); 
    ax = Axis(f[1,1], xlabel="kyrs", ylabel="Stress Mpa")
    scatter!(ax, t./1e3,  sol_τyy./1e6, label="analytic", linewidth=3)
    lines!(ax, t./1e3,  av_τyy./1e6, label="numeric", linewidth=3, color = :black)
    axislegend(ax)
    ylims!(ax, 0, 220)
    f
end

function elastic_buildup(; nx=256-1, ny=256-1, lx=100e3, ly=100e3, endtime = 500, η0 = 1e22, εbg = 1e-14, G = 10^10)
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
    yr = 365.25*3600*24
    Myr, kyr = 1e6*yr, 1e3*yr
    ttot = endtime*kyr # total simulation time

    ## Setup-specific parameters and fields
    η = [η0*(1 + (rand()-0.5)*0.001) for _ in 1:nx, _ in 1:ny] 
    g = 0.0 # gravity

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg) 
    freeslip = (freeslip_x = true, freeslip_y = true)

    # Physical time loop
    t = 0.0
    ρ = @zeros(size(stokes.P))
    local iters
    av_τyy, sol_τyy, tt = Float64[], Float64[], Float64[]
    while t < ttot
        if t<5*kyr
            dt = 0.1*kyr
        else
            dt = 2*kyr
        end
        iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, ρ.*g, η, G, dt; iterMax = 10e3)
        t += dt

        push!(av_τyy,  mean(stokes.τ.yy))
        push!(sol_τyy, solution(εbg, t, G, η0))
        push!(tt, t/yr)
    end

    return  (ni=ni, xci=xci, xvi=xvi, li=li), stokes, av_τyy, sol_τyy, tt, iters

end

function multiple_elastic_buildup(; lx=100e3, ly=100e3, endtime = 500, η0 = 1e22, εbg = 1e-14, G = 10^10, nrange::UnitRange = 4:8)
    
    av_err = Float64[]  
    for i in nrange
        nx = ny = 2^i-1
        geometry, stokes, av_τyy, sol_τyy, t, iters = 
            elastic_buildup(nx=nx, ny=ny, lx=lx, ly=ly, endtime = endtime, η0 = η0, εbg = εbg, G = G)
       
        push!(av_err, mean(@. abs(av_τyy - sol_τyy)/sol_τyy))
    end

    nx = @. 2^nrange-1
    h = @. (1/nx)

    f = Figure(fontsize=28) 
    ax = Axis(f[1,1], yscale = log10, xscale = log10, yminorticksvisible = true, yminorticks = IntervalsBetween(8))
    lines!(ax, h, av_err, linewidth=3)
    ax.xlabel = "h"
    ax.ylabel = "error ||av_τyy - sol_τyy||/sol_τyy"
    f

end