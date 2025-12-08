using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, width, height, fault_thickness, fault_dip)
    ni      = size(phase_ratios.center)
    lx      = xvi[1][end] - xvi[1][1]

    fault   = GGU.Rectangle((0e0, height / 2), Inf, fault_thickness; θ = deg2rad(fault_dip))
    sample  = GGU.Rectangle((0e0, height / 2), width, height)
    buffer1 = GGU.Rectangle((xvi[1][1]   + buffer/2, height / 2), buffer, height)
    buffer2 = GGU.Rectangle((xvi[1][end] - buffer/2, height / 2), buffer, height)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, sample, fault, buffer1, buffer2)
        x, y     = xc[i], yc[j]
        p        = GGU.Point(x, y)
        infault  = GGU.inside(p, fault)
        insample = GGU.inside(p, sample)
        inbuffer = GGU.inside(p, buffer1) || GGU.inside(p, buffer2)
        
        if insample
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 0.0
        end
        if infault
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
            @index phases[3, i, j] = 0.0
        end
        if inbuffer
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 0.0
            @index phases[3, i, j] = 1.0
        end
        
        
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., sample, fault, buffer1, buffer2)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., sample, fault, buffer1, buffer2)
    return nothing
end

 # domain
 height          = 3e0
    width           = height / 3
    buffer          = 0.2
    fault_thickness = 0.1
    fault_dip       = 45


# # Physical domain ------------------------------------
# lx = width + 2 * buffer  # domain length in x
# ly = height              # domain length in y
# ni = nx, ny              # number of cells
# li = lx, ly              # domain length in x- and y-
# di = @. li / ni          # grid step in x- and -y
# origin = -lx / 2, 0.0    # origin coordinates
# grid = Geometry(ni, li; origin = origin)
# (; xci, xvi) = grid # nodes at the center and vertices of the cells

# # Initialize phase ratios -------------------------------
# phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
# init_phases!(phase_ratios, xci, xvi, width, height, fault_thickness, fault_dip)

# p = [argmax(p) for p in phase_ratios.center]
# display(heatmap(p))

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # domain
    height          = 3e0
    width           = height / 3
    buffer          = 0.1
    fault_thickness = 0.1
    fault_dip       = 45

    # Physical domain ------------------------------------
    lx = width + 2 * buffer  # domain length in x
    ly = height              # domain length in y
    ni = nx, ny              # number of cells
    li = lx, ly              # domain length in x- and y-
    di = @. li / ni          # grid step in x- and -y
    origin = -lx / 2, 0.0    # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = 1e-2
    # Physical properties using GeoParams ----------------
    rheology = (
        # host rock
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (
                    PowerlawViscous(; η0 = 1e18, n=1.0, ε0=1),
                    ConstantElasticity(; G = 10e9, Kb = 1e11),
                    DruckerPrager_regularised(;
                        C = 15e6,    # cohesion
                        ϕ = 35,      # friction angle
                        η_vp = 1e15/3, # regularization
                        Ψ = 0,       # dilation angle
                    ),
                )
            ),
            Elasticity = ConstantElasticity(; G = 10e9, Kb = 1e11),
        ),
        # fault gauge
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (
                    ConstantElasticity(; G = 1e9, Kb = 1e10),
                    PowerlawViscous(; η0 = 1e18, n=1.0, ε0=1),
                    DruckerPrager_regularised(;
                        C = 1e6,     # cohesion
                        ϕ = 30,      # friction angle
                        η_vp = 1e15/3, # regularization
                        Ψ = 0,       # dilation angle
                    ),
                )
            ),
            Elasticity = ConstantElasticity(; G = 1e9, Kb = 1e10),
        ),
        # buffer material
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (
                    LinearViscous(; η = 1e12),
                    ConstantElasticity(; G = 10e9, Kb = 1e11),
                )    
            ),
            Elasticity = ConstantElasticity(; G = 10e9, Kb = 1e11),
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, width, height, fault_thickness, fault_dip)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-3, CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
        # free_slip = (left = true, right = true, top = false, bot = false),
        # no_slip = (left = false, right = false, top = true, bot = true),
    )
    εbg = 1e-6
    stokes.V.Vx .= PTArray(backend)([x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]
    # while t < tmax
    for _ in 1:50

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                verbose = false,
                iterMax = 15.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t  += dt

        println("it = $it; t = $t \n")

        push!(τII, stokes.τ.II[ni[1]>>>1, end])
        # visualisation
        
        ar = lx/ly
        fig = Figure(size = (800, 800), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = ar, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = ar, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = ar, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2])
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        scatterlines!(ax4, τII, color = :blue)
        # heatmap!(ax4, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
       
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        display(fig)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

n = 128
nx = n
ny = n
figdir = "ShearBands2D"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
