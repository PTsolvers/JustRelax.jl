push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test#, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

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
function ShearBand2D()
    n = 32
    nx = n
    ny = n
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)


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
    τ_y = 1.6            # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30             # friction angle
    C = τ_y            # Cohesion
    η0 = 1.0            # viscosity
    G0 = 1.0            # elastic shear modulus
    Gi = G0 / 2         # elastic shear modulus perturbation
    εbg = 1.0            # background strain-rate
    η_reg = 1.0e-2         # regularisation "viscosity"
    dt = η0 / G0 / 4.0  # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 5)
    el_inc = ConstantElasticity(; G = Gi, Kb = 5)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0
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
    stokes = StokesArrays(backend_JR, ni)

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
    )
    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0.0e0
    @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0.0e0
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # Time loop
    t, it = 0.0, 0
    τII = [0.0e0]
    sol = [0.0e0]
    ttot = [0.0e0]
    local iters, τII, sol

    while it < 10

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
                # verbose_PH = false,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 50,
                rel_drop = 0.5,
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

    end
    tensor_invariant!(stokes.τ)

    finalize_global_grid(; finalize_MPI = true)

    return iters, τII, sol, extrema(stokes.τ.II)
end

@testset "ShearBand2D" begin
    # @suppress begin
        iters, τII, sol, extrema_τII = ShearBand2D()
        @test iters.err_evo_tot[end] < 1.0e-6
        @test extrema_τII[1] ≈ 1.544 atol = 1.0e-3
        @test extrema_τII[2] ≈ 1.639 atol = 1.0e-3
        @test τII[end] ≈ 1.6388 atol = 1.0e-4
        @test sol[end] ≈ 1.8358 atol = 1.0e-4
    # end
end
