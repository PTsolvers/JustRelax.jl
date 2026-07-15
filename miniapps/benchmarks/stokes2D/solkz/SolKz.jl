using Pkg; Pkg.activate("miniapps")
using ParallelStencil.FiniteDifferences2D # this is needed because the viscosity and density functions live outside JustRelax scope
using JLD2

# include benchmark related plotting and error functions
include("vizSolKz.jl")

function solKz_viscosity(xci, ni, di; B = log(1.0e6))
    xc, yc = xci
    # make grid array (will be eaten by GC)
    y = @zeros(ni...)
    y = PTArray(backend)([yci for _ in xc, yci in yc])
    η = @zeros(ni...)

    _viscosity(y, B) = exp(B * y)

    @parallel function viscosity(η, y, B)

        @all(η) = _viscosity(@all(y), B)
        return nothing
    end

    # compute viscosity
    @parallel viscosity(η, y, B)

    return η
end

function solKz_density(xci, ni, di; km = 2)
    xc, yc = xci
    # make grid array (will be eaten by GC)
    x = PTArray(backend)([xci for xci in xc, _ in yc])
    y = PTArray(backend)([yci for _ in xc, yci in yc])
    ρ = @zeros(ni...)

    # matches Stokes2D_SolKz_Zhong1996's reference density: ρ = -σ*sin(km*y)*cos(kn*x), σ=1, kn=3π
    _density(x, y, km) = -sin(km * y) * cos(3 * π * x)

    @parallel function density(ρ, x, y, km)

        @all(ρ) = _density(@all(x), @all(y), km)
        return nothing
    end

    # compute density
    @parallel density(ρ, x, y, km)

    return ρ
end

function solKz(;
        Δη = 1.0e6, km = 2, nx = 256 - 1, ny = 256 - 1, lx = 1.0e0, ly = 1.0e0, init_MPI = true, finalize_MPI = false
    )

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = nx, ny  # number of nodes in x- and y-
    li = lx, ly  # domain length in x- and y-
    origin = zero(nx), zero(ny)
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...) #init MPI
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    g = 1 # gravity

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1 # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(backend, ni)
    (; η) = stokes.viscosity
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; Re = 5π, CFL = 1 / √2.1)

    ## Setup-specific parameters and fields
    η .= solKz_viscosity(xci, ni, di; B = log(Δη)) # viscosity field
    ρ = solKz_density(xci, ni, di; km = km)
    fy = ρ .* g
    ρg = @zeros(ni...), fy
    dt = 0.1
    G = @fill(Inf, ni...)
    Kb = @fill(Inf, ni...)

    ## Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true)
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Physical time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve!(
            stokes,
            pt_stokes,
            grid,
            flow_bcs,
            ρg,
            G,
            Kb,
            dt,
            igg;
            kwargs = (
                iterMax = 150.0e3,
                nout = 1.0e3,
                b_width = (4, 4, 0),
                verbose = true,
            ),
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters, ρ
end

function multiple_solKz(; Δη = 1.0e6, km = 2, nrange::UnitRange = 4:10)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solKz(; Δη = Δη, km = km, nx = nx, ny = ny, init_MPI = false, finalize_MPI = false)
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes; order = 1, Δη = Δη, km = km)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
    end

    nx = @. 2^nrange - 1
    h = @. (1 / nx)

    f = Figure(; fontsize = 28)
    ax = Axis(
        f[1, 1];
        yscale = log10,
        xscale = log10,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(8),
    )
    lines!(ax, h, (L2_vx); linewidth = 3, label = "Vx")
    lines!(ax, h, (L2_vy); linewidth = 3, label = "Vy")
    lines!(ax, h, (L2_p); linewidth = 3, label = "P")
    axislegend(ax; position = :rt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save("SolKz_error.png", f)

    jldsave(joinpath(@__DIR__, "solkz_normal_error.jld2"); h, L2_vx, L2_vy, L2_p, Δη, km, nrange)

    return f
end
