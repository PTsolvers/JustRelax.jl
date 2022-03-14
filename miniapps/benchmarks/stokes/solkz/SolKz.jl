using ParallelStencil.FiniteDifferences2D # this is needed because the viscosity and density functions live outside JustRelax scope

# include benchmark related plotting and error functions
include("vizSolKz.jl")

function solKz_viscosity(xci, ni; B=log(1e6))
    xc, yc = xci
    # make grid array (will be eaten by GC)
    y = PTArray([yci for _ in xc, yci in yc])
    η = @zeros(ni...)
    # inner closure
    _viscosity(y, B) = exp(B * y)
    # outer closure
    @parallel function viscosity(η, y, B)
        @all(η) = _viscosity(@all(y), B)
        return nothing
    end
    # compute viscosity
    @parallel viscosity(η, y, B)

    return η
end

function solKz_density(xci, ni)
    xc, yc = xci
    # make grid array (will be eaten by GC)
    x = PTArray([xci for xci in xc, _ in yc])
    y = PTArray([yci for _ in xc, yci in yc])
    ρ = @zeros(ni...)
    # inner closure
    _density(x, y) = -sin(2 * y) * cos(3 * π * x)
    # outer closure
    @parallel function density(ρ, x, y)
        @all(ρ) = _density(@all(x), @all(y))
        return nothing
    end
    # compute density
    @parallel density(ρ, x, y)

    return ρ
end

function solKz(; Δη=1e6, nx=256 - 1, ny=256 - 1, lx=1e0, ly=1e0)

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    xci = Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells
    g = 1 # gravity

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, Viscous)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    η = solKz_viscosity(xci, ni; B=log(Δη)) # viscosity field
    ρ = solKz_density(xci, ni)
    fy = ρ * g

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true)

    # Physical time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, fy, η; iterMax=10e3)
        t += Δt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters, ρ
end

function multiple_solKz(; Δη=1e-6, nrange::UnitRange=4:10)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solKz(; Δη=Δη, nx=nx, ny=ny)
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes; order=1)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
    end

    nx = @. 2^(nrange) - 1
    h = @. (1 / nx)

    f = Figure(; fontsize=28)
    ax = Axis(
        f[1, 1];
        yscale=log10,
        xscale=log10,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(8),
    )
    lines!(ax, h, (L2_vx); linewidth=3, label="Vx")
    lines!(ax, h, (L2_vy); linewidth=3, label="Vy")
    lines!(ax, h, (L2_p); linewidth=3, label="P")
    axislegend(ax; position=:rt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save("SolKz_error.png", f)

    return f
end
