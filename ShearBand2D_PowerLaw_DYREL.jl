using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
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
    origin = @. -li / 2  # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt  = Inf
    εbg = 1
    
    # Physical properties using GeoParams ----------------
    visc_bg  = PowerlawViscous(; η0 = 1e2,  n=3, ε0 = 1e0)
    visc_inc = PowerlawViscous(; η0 = 1e-1, n=3, ε0 = 1e0)

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_bg,)),
        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_inc,)),
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    origin = 0.0, 0.0
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = Inf)

    # Rheology
    stokes.ε.xx   .= 1
    stokes.ε.xx_v .= 1
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

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, di, Inf; ϵ=1e-6);
    1 
    # Time loop
    t, it = 0.0, 0

    # Stokes solver ----------------
    solve_DYREL!(
        stokes,
        ρg,
        dyrel,
        flow_bcs,
        phase_ratios,
        rheology,
        args,
        di,
        dt,
        igg;
        kwargs = (;
            verbose  = false,
            iterMax  = 50.0e3,
            nout     = 400,
            rel_drop = 1e-5,
            # λ_relaxation = 0,
            λ_relaxation_DR = 1,
            λ_relaxation_PH = 1,
            viscosity_relaxation = 1e-1,
            viscosity_cutoff = (-Inf, Inf),
        )
    );
    tensor_invariant!(stokes.ε)
    tensor_invariant!(stokes.ε_pl)

    it += 1
    t += dt

    println("it = $it; t = $t \n")
    
    # visualisation
    th = 0:(pi / 50):(3 * pi)
    xunit = @. radius * cos(th) + 0.5
    yunit = @. radius * sin(th) + 0.5
    fig = Figure(size = (800, 800), title = "t = $t")
    ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
    ax2 = Axis(fig[2, 1], aspect = 1, title = L"\eta^{\text{eff}}", titlesize = 35)
    ax3 = Axis(fig[1, 3], aspect = 1, title = L"\varepsilon_{II}", titlesize = 35)
    ax4 = Axis(fig[2, 3], aspect = 1)
    h11 = GLMakie.heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
    # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
    h21 = GLMakie.heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
    h22 = GLMakie.heatmap!(ax4, xci..., Array(log10.(stokes.viscosity.η)), colormap = :batlow)
    # h21 = heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
    h12 = GLMakie.heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
    # lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
    # lines!(ax4, ttot, τII, color = :black)
    # lines!(ax4, ttot, sol, color = :red)
    Colorbar(fig[1, 2], h11)
    Colorbar(fig[2, 2], h21)
    Colorbar(fig[1, 4], h12)
    Colorbar(fig[2, 4], h22)
    hidexdecorations!(ax1)
    hidexdecorations!(ax3)
    display(fig)

    save(joinpath(figdir, "$(it).png"), fig)

    return nothing
end

n  = 128
nx = n
ny = n
figdir = "ShearBands2D_PowerLaw_DYREL"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);