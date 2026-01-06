const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using JustPIC, JustPIC._2D
const backend = @static if isCUDA
    JustPIC.CUDABackend
else
    JustPIC.CPUBackend
end

using GeoParams, CairoMakie

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases)
        @index phases[1, i, j] = 1.0

        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex)
    return nothing
end

const yr = 365.25 * 3600 * 24
const kyr = 1.0e3 * yr

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 100.0e3        # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    ## (Physical) Time domain and discretization
    endtime = 500
    ttot = endtime * kyr # total simulation time

    # Physical properties using GeoParams ----------------
    η0 = 1.0e22  # viscosity
    G0 = 1.0e10    # elastic shear modulus
    εbg = 1.0e-14 # background strain-rate
    dt = 1.0e11
    el_bg = SetConstantElasticity(; G = G0, ν = 0.49)
    visc = LinearViscous(; η = η0)
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0e0),
            Gravity = ConstantGravity(; g = 0.0e0),
            CompositeRheology = CompositeRheology((visc, el_bg)),
            Elasticity = el_bg,
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(phase_ratios)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.75 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = Inf)

    # Rheology
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
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

        dt = t < 10 * kyr ? 0.05 * kyr : 1.0 * kyr

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
                iterMax = 100.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
                relaxation = 1.0e-2,
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
        lines!(ax1, ttot, τII ./ 1.0e6; color = :black, label = "τII")
        lines!(ax1, ttot, sol ./ 1.0e6; color = :red, label = "sol")
        Legend(fig[1, 2], ax1)
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
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
