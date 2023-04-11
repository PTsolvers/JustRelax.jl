import Statistics: mean

# Analytical solution
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

function plot_elastic_buildup(av_τyy, sol_τyy, t)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel="kyrs", ylabel="Stress Mpa")
    scatter!(ax, t ./ 1e3, sol_τyy ./ 1e6; label="analytic", linewidth=3)
    lines!(ax, t ./ 1e3, av_τyy ./ 1e6; label="numeric", linewidth=3, color=:black)
    axislegend(ax)
    ylims!(ax, 0, 220)
    return f
end

function elastic_buildup(;
    nx=256 - 1, ny=256 - 1, lx=100e3, ly=100e3, endtime=500, η0=1e22, εbg=1e-14, G=10^10
)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    nDim = length(ni) # domain dimension
    origin = 0.0, 0.0
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    yr = 365.25 * 3600 * 24
    kyr = 1e3 * yr
    ttot = endtime * kyr # total simulation time

    ## Setup-specific parameters and fields
    η = fill(η0, nx, ny)
    g = 0.0 # gravity
    Gc = @fill(G, ni...)
    Kb = @fill(Inf, ni...)

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6, CFL=1 / √2.1)

    ## Boundary conditions
    pureshear_bc!(stokes, xci, xvi, εbg)
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    flow_bcs!(stokes, flow_bcs, di)

    # Physical time loop
    t = 0.0
    ρg = @zeros(ni...), @ones(size(stokes.P)) .* g
    local iters
    av_τyy, sol_τyy, tt = Float64[], Float64[], Float64[]
    while t < ttot
        dt = t < 10 * kyr ? 0.05 * kyr : 1.0 * kyr
        iters = solve!(
            stokes, pt_stokes, di, flow_bcs, ρg, η, Gc, Kb, dt; iterMax=150e3, nout=1000
        )

        @show t += dt

        push!(av_τyy, maximum(abs.(stokes.τ.yy)))
        # push!(av_τyy, maximum(stokes.τ.yy))
        push!(sol_τyy, solution(εbg, t, G, η0))
        push!(tt, t / kyr)
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li), stokes, av_τyy, sol_τyy, tt, iters
end

function multiple_elastic_buildup(;
    lx=100e3, ly=100e3, endtime=500, η0=1e22, εbg=1e-14, G=10^10, nrange::UnitRange=4:8
)
    av_err = Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, av_τyy, sol_τyy, t, iters = elastic_buildup(;
            nx=nx, ny=ny, lx=lx, ly=ly, endtime=endtime, η0=η0, εbg=εbg, G=G
        )

        push!(av_err, mean(@. abs(av_τyy - sol_τyy) / sol_τyy))
    end

    nx = @. 2^nrange - 1
    h = @. (1 / nx)

    f = Figure(; fontsize=28)
    ax = Axis(
        f[1, 1];
        yscale=log10,
        xscale=log10,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(8),
    )
    lines!(ax, h, av_err; linewidth=3)
    ax.xlabel = "h"
    ax.ylabel = "error ||av_τyy - sol_τyy||/sol_τyy"

    save("ElasticBuildUp.png", f)

    return f
end
