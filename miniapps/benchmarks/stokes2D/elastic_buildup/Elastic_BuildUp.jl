import Statistics: mean

# Analytical solution
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

function plot_elastic_buildup(av_τyy, sol_τyy, t)
    f = Figure()
    ax = Axis(f[1, 1]; xlabel = "kyrs", ylabel = "Stress Mpa")
    scatter!(ax, t ./ 1.0e3, sol_τyy ./ 1.0e6; label = "analytic", linewidth = 3)
    lines!(ax, t ./ 1.0e3, av_τyy ./ 1.0e6; label = "numeric", linewidth = 3, color = :black)
    axislegend(ax)
    ylims!(ax, 0, 220)
    return f
end

function elastic_buildup(;
        nx = 256 - 1,
        ny = 256 - 1,
        lx = 100.0e3,
        ly = 100.0e3,
        endtime = 500,
        η0 = 1.0e22,
        εbg = 1.0e-14,
        G = 10^10,
        init_MPI = false,
        finalize_MPI = false,
    )
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = nx, ny # number of nodes in x- and y-
    li = lx, ly  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...) # init MPI
    origin = 0.0, 0.0
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    yr = 365.25 * 3600 * 24
    kyr = 1.0e3 * yr
    ttot = endtime * kyr # total simulation time

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, CFL = 1 / √2.1)

    ## Setup-specific parameters and fields
    (; η) = stokes.viscosity
    η .= @fill(η0, ni...)
    g = 0.0 # gravity
    Gc = @fill(G, ni...)
    Kb = @fill(Inf, ni...)

    ## Boundary conditions
    pureshear_bc!(stokes, xci, xvi, εbg, backend)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true)
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # Physical time loop
    t = 0.0
    it = 0
    ρg = @zeros(ni...), @ones(size(stokes.P)) .* g
    av_τyy = Float64[]
    sol_τyy = Float64[]
    tt = Float64[]
    local iters
    while t < ttot
        dt = t < 10 * kyr ? 0.05 * kyr : 1.0 * kyr
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            Gc,
            Kb,
            dt,
            igg;
            kwargs = (;
                iterMax = 150.0e3,
                nout = 1000,
                b_width = (4, 4, 0),
                verbose = true,
            )
        )

        t += dt
        it += 1
        println("Iteration $it => t = $(t / kyr) kyrs")

        push!(av_τyy, maximum(abs.(stokes.τ.yy)))
        push!(sol_τyy, solution(εbg, t, G, η0))
        push!(tt, t / kyr)
    end
    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li), stokes, av_τyy, sol_τyy, tt, iters
end

function multiple_elastic_buildup(;
        lx = 100.0e3, ly = 100.0e3, endtime = 500, η0 = 1.0e22, εbg = 1.0e-14, G = 10^10, nrange::UnitRange = 4:8
    )
    av_err = Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        _, _, av_τyy, sol_τyy, = elastic_buildup(;
            nx = nx,
            ny = ny,
            lx = lx,
            ly = ly,
            endtime = endtime,
            η0 = η0,
            εbg = εbg,
            G = G,
            init_MPI = false,
            finalize_MPI = false,
        )

        push!(av_err, mean(@. abs(av_τyy - sol_τyy) / sol_τyy))
    end

    nx = @. 2^nrange - 1
    h = @. 1 / nx
    f = Figure(; fontsize = 28)
    ax = Axis(
        f[1, 1];
        yscale = log10,
        xscale = log10,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(8),
    )
    lines!(ax, h, av_err; linewidth = 3)
    ax.xlabel = "h"
    ax.ylabel = "error ||av_τyy - sol_τyy||/sol_τyy"

    save("ElasticBuildUp.png", f)

    return f
end
