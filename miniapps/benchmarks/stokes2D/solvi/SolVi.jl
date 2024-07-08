# include benchmark related functions
include("vizSolVi.jl")

@parallel function smooth!(
    A2::AbstractArray{T,2}, A::AbstractArray{T,2}, fact::Real
) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

function _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    for i in eachindex(xci), j in eachindex(yci)
        if rc < sqrt((xci[i] - cx)^2 + (yci[j] - cy)^2)
            η[i, j] = ηi
        end
    end
end

function solvi_viscosity(ni, di, li, rc, η0, ηi)
    dx, dy = di
    lx, ly = li
    η      = @fill(η0, ni...)
    Rad2   = [
        sqrt.(
            ((ix - 1) * dx + 0.5 * dx - 0.5 * lx)^2 +
            ((iy - 1) * dy + 0.5 * dy - 0.5 * ly)^2,
        ) for ix in 1:ni[1], iy in 1:ni[2]
    ]
    η[Rad2 .< rc] .= ηi

    return η
end

function solVi(;
    Δη=1e-3,
    nx=256 - 1,
    ny=256 - 1,
    lx=1e1,
    ly=1e1,
    rc=1e0,
    εbg=1e0,
    init_MPI=true,
    finalize_MPI=false,
)
    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni           = (nx, ny) # number of nodes in x- and y-
    li           = (lx, ly)  # domain length in x- and y-
    origin       = zero(nx), zero(ny)
    igg          = IGG(init_global_grid(nx, ny, 1; init_MPI=init_MPI)...) #init MPI
    di           = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    ttot      = 1 # total simulation time
    Δt        = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes    = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-5, CFL=0.27 / √2.1)

    ## Setup-specific parameters and fields
    η0        = 1.0  # matrix viscosity
    ηi        = Δη # inclusion viscosity
    η         = solvi_viscosity(ni, di, li, rc, η0, ηi) # viscosity field
    ρg        = @zeros(ni...), @zeros(ni...)
    dt        = 0.1
    G         = @fill(Inf, ni...)
    Kb        = @fill(Inf, ni...)

    ## Boundary conditions
    pureshear_bc!(stokes, xci, xvi, εbg)
    flow_bcs  = VelocityBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true)
    )
    flow_bcs!(stokes,flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Physical time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            G,
            Kb,
            dt,
            igg;
            iterMax = 150e3,
            nout    = 1e3,
            b_width = (4, 4, 1),
            verbose = false,
        )
        t += Δt
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

function multiple_solVi(; Δη=1e-3, lx=1e1, ly=1e1, rc=1e0, εbg=1e0, nrange::UnitRange=4:8)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, iters = solVi(;
            Δη           = Δη,
            nx           = nx,
            ny           = ny,
            lx           = lx,
            ly           = ly,
            rc           = rc,
            εbg          = εbg,
            init_MPI     = false,
            finalize_MPI = false,
        )
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes, Δη, εbg, rc; order = 2)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
    end

    nx = @. 2^nrange - 1
    h  = @. (1 / nx)
    f  = Figure(; fontsize=28)
    ax = Axis(
        f[1, 1];
        yscale=log10,
        xscale=log10,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(8),
    )
    lines!(ax, h, (L2_vx); linewidth = 3, label="Vx")
    lines!(ax, h, (L2_vy); linewidth = 3, label="Vy")
    lines!(ax, h, (L2_p);  linewidth = 3, label="P")
    axislegend(ax; position=:lt)
    ax.xlabel = "h"
    ax.ylabel = "L2 norm"

    save("SolVi_error.png", f)

    return f
end
