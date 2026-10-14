using JustRelax, JustRelax.JustRelax2D

using GeoParams
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC
import JustPIC.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPU

using Pkg; Pkg.activate("miniapps")
using CairoMakie

# HELPER FUNCTIONS ----------------------------------- ----------------------------
@inline function tensile_cap_params(sinϕ::T, cosϕ::T, sinψ::T, C::T, pT::T) where {T}
    ps = -pT               # tensile limit on tension-positive axis
    k = sinϕ
    kf = sinψ
    c = C * cosϕ

    a = sqrt(one(T) + k * k)
    cosa = inv(a)
    sina = k * cosa

    py = (ps + c * cosa) / (one(T) - sina)
    R = py - ps

    pd = py - R * sina
    sd = c + k * pd

    pf = pd + kf * (c + k * pd)
    b = sqrt(one(T) + kf * kf)
    Rf = pf - ps

    norm_pf = hypot(pd - pf, sd)
    pdf = pf + Rf * (pd - pf) / norm_pf
    sdf = Rf * sd / norm_pf

    return (; k, kf, c, a, b, pd, sd, py, R, pf, Rf, pdf, sdf)
end

function draw_yield_surface!(ax, cp, xmax, xc, yc)
    # Using names from tensile_cap_params
    py, pd, sd, c, k = cp.py, cp.pd, cp.sd, cp.c, cp.k
    hlines!(ax, 0; color = :black, linewidth = 1)
    vlines!(ax, 0; color = :black, linewidth = 1)
    lines!(ax, [py, pd], [0.0, sd]; color = :red, linestyle = :dash, linewidth = 1.5)
    lines!(ax, [pd, xmax], [sd, c + k * xmax]; color = :red, linewidth = 2)
    return lines!(ax, xc, yc; color = :red, linewidth = 2)
end

solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
        if GGU.inside(p, circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0

        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
end


# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly             # domain length in x
    ni = nx, ny         # number of cells
    li = lx, ly         # domain length in x- and y-
    di = @. li / ni     # grid step in x- and -y
    origin = 0.0, 0.0       # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid           # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress (cohesion: c*cos(ϕ))
    ϕ = 30              # friction angle
    ψ = 5               # Dilation angle to induce negative pressures (triggers Cap)
    C = τ_y             # Cohesion
    η0 = 1.0            # viscosity
    G0 = 1.0            # elastic shear modulus
    Gi = G0 / 2.0       # softer inclusion
    εbg = 1.0           # background strain-rate in x (shear component)
    η_reg = 1.0e-3      # regularisation "viscosity"
    dt = η0 / G0 / 8.0  # decreased dt to stabilize transition

    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    # Cap plasticity. mode1 = DP, mode2 = Cap
    pl = DruckerPragerCap(;
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = ψ,          # Dilation causes pressure drop
        pT = -0.5,      # Tensile cap limit
        # softening_C = soft_C
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    origin = 0.5, 0.5
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0e0
    @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0e0
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # Time loop
    t, it = 0.0, 0
    τII = [0.0e0]
    sol = [0.0e0]
    ttot = [0.0e0]

    for _ in 1:150

        # Stokes solver ----------------
        iters = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 10,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1,
                linear_viscosity = true,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        it += 1
        t += dt

        push!(τII, maximum(stokes.τ.xx))
        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # Visualisation
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5

        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = "Pressure (P)", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1, title = L"P \text{ vs } \tau_{II}", titlesize = 35)
        ax5 = Axis(fig[1, 3], aspect = 1, title = L"E_{Vol}^{II}", titlesize = 35)
        ax6 = Axis(fig[2, 3], aspect = 1, title = "Stress over time", titlesize = 35)
        # Plot Pressure to show negative excursions
        heatmap!(ax1, xci..., Array(stokes.P), colormap = :vik, colorrange = (-1.0, 1.0))
        heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        heatmap!(ax5, xci..., Array(stokes.EVol_pl), colormap = :batlow)

        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)

        # ax4 plotting
        cp = tensile_cap_params(sind(ϕ), cosd(ϕ), sind(ψ), C / cosd(ϕ), abs(pl.pT.val))
        xc_array = range(-abs(pl.pT.val), cp.pd; length = 100)
        yc_array = sqrt.(max.(0.0, cp.R^2 .- (collect(xc_array) .- cp.py) .^ 2))

        P_pts = vec(Array(stokes.P))
        τII_pts = vec(Array(stokes.τ.II))
        xmax_plot = maximum(P_pts) + 0.5
        draw_yield_surface!(ax4, cp, max(xmax_plot, 2.0), xc_array, yc_array)
        scatter!(ax4, P_pts, τII_pts; color = (:blue, 0.5), markersize = 3)

        lines!(ax6, ttot, τII, color = :black, label = "Numerical")
        lines!(ax6, ttot, sol, color = :red, label = "Analytical")
        axislegend(ax6, position = :rb)

        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end


    return nothing
end

nx = 128
ny = 128
figdir = "ShearBands2D_DYREL"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
@time main(igg; figdir = figdir, nx = nx, ny = ny);
