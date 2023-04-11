using ParallelStencil.FiniteDifferences2D # this is needed because the viscosity and density functions live outside JustRelax scope

# include plotting and error related functions
include("vizSolCx.jl")

@parallel function smooth!(
    A2::AbstractArray{T,2}, A::AbstractArray{T,2}, fact::Real
) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

function solCx_viscosity(xci, ni; Δη=1e6)
    xc, yc = xci
    # make grid array (will be eaten by GC)
    x = PTArray([xci for xci in xc, _ in yc])
    η = PTArray(zeros(ni...))
    # inner closure
    _viscosity(x, Δη) = ifelse(x ≤ 0.5, 1e0, Δη)
    # outer closure
    @parallel function viscosity(η, x)
        @all(η) = _viscosity(@all(x), Δη)
        return nothing
    end
    # compute viscosity
    @parallel viscosity(η, x)

    return η
end

function solCx_density(xci, ni)
    xc, yc = xci
    # make grid array (will be eaten by GC)
    x = PTArray([xci for xci in xc, _ in yc])
    y = PTArray([yci for _ in xc, yci in yc])
    ρ = PTArray(zeros(ni))
    # inner closure
    _density(x, y) = -sin(π * y) * cos(π * x)
    # outer closure
    @parallel function density(ρ, x, y)
        @all(ρ) = _density(@all(x), @all(y))
        return nothing
    end
    # compute viscosity
    @parallel density(ρ, x, y)

    return ρ
end

function solCx(Δη=Δη; nx=256 - 1, ny=256 - 1, lx=1e0, ly=1e0, init_MPI=true, finalize_MPI=false)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    origin = zero(nx), zero(ny)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI=init_MPI)...) #init MPI
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells

    # di = @. li / ni # grid step in x- and -y
    # max_li = max(li...)
    # nDim = length(ni) # domain dimension
    # xci = Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]) # nodes at the center of the cells
    # xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells
    g = 1

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; CFL=1 / √2.1)

    ## Setup-specific parameters and fields
    η = solCx_viscosity(xci, ni; Δη=Δη) # viscosity field
    ρ = solCx_density(xci, ni)
    fy = ρ .* g
    ρg = @zeros(ni...), fy
    dt = Inf
    G = @fill(Inf, ni...)
    K = @fill(Inf, ni...)

    # smooth viscosity jump (otherwise no convergence for Δη > ~15)
    η2 = deepcopy(η)
    for _ in 1:5
        @parallel smooth!(η2, η, 1.0)
        @parallel (1:size(η2, 1)) free_slip_y!(η2)
        η, η2 = η2, η
    end

    ## Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    # Physical time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve!(
            stokes, pt_stokes, di, flow_bcs, ρg, η, G, K, dt, igg::IGG;
            iterMax=10e3,
            nout=500,
            b_width=(4,4,1)
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters, ρ
end

function multiple_solCx(; Δη=1e6, nrange::UnitRange=6:10)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solCx(Δη; nx=nx, ny=ny)
        L2_vxi, L2_vyi, L2_pi = solcx_error(geometry, stokes; order=1)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
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
    lines!(ax, h, (L2_vx); linewidth=3, label="Vx")
    lines!(ax, h, (L2_vy); linewidth=3, label="Vy")
    lines!(ax, h, (L2_p); linewidth=3, label="P")
    axislegend(ax; position=:lt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save("SolCx_error.png", f)

    return f
end
