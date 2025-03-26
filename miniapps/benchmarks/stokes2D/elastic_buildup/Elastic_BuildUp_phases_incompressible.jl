using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using GeoParams, GLMakie, CellArrays

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases)
        @index phases[1, i, j] = 1.0

        return nothing
    end

    return @parallel (@idx ni) init_phases!(phase_ratios.center)
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = Inf

    # Physical properties using GeoParams ----------------
    η0 = 1.0e22           # viscosity
    G0 = 10^10           # elastic shear modulus
    εbg = 1.0e-14           # background strain-rate
    dt = 1.0e11
    el_bg = SetConstantElasticity(; G = G0, ν = 0.5)
    visc = LinearViscous(; η = η0)
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2700.0),
            Gravity = ConstantGravity(; g = 9.81),
            CompositeRheology = CompositeRheology((visc, el_bg)),
            Elasticity = el_bg,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius = 0.1
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(phase_ratios)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, CFL = 0.75 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    ρg[2] .= rheology[1].Density[1].ρ.val .* rheology[1].Gravity[1].g.val
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend_JR)([x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    tmax = 1.0e13
    τII = Float64[0.0]
    sol = Float64[0.0]
    ttot = Float64[0.0]
    P = Float64[0.0]

    while t < tmax

        # Stokes solver ----------------
        solve!(
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
                iterMax = 500.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        push!(τII, maximum(stokes.τ.xx))


        it += 1
        t += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)
        push!(P, maximum(stokes.P))

        println("it = $it; t = $t \n")

        fig = Figure(; size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1]; aspect = 1, title = "τII")
        ax2 = Axis(fig[2, 1]; aspect = 1, title = "Pressure")
        lines!(ax1, ttot, τII ./ 1.0e6; color = :black, label = "τII")
        lines!(ax1, ttot, sol ./ 1.0e6; color = :red, label = "sol")
        lines!(ax2, ttot, P; color = :black, label = "P")
        Legend(fig[1, 2], ax1)
        Legend(fig[2, 2], ax2)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

N = 32
n = N + 2
nx = n - 2
ny = n - 2
figdir = "ElasticBuildUp_incompressible"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
