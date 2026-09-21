const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")

const backend = @static if isCUDA
    JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using GeoParams, GLMakie


import JustPIC.GridGeometryUtils as GGU


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
function main(igg; nx = 64, ny = 64, figdir = "model_figs", nsteps = 30)

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
    τ_y = 1.6              # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30                 # friction angle
    C = τ_y                # Cohesion
    η0 = 1.0               # viscosity
    G0 = 1.0               # elastic shear modulus
    Gi = G0 / 2            # elastic shear modulus perturbation
    εbg = 1.0              # background strain-rate
    η_reg = 1.0e-2         # regularisation "viscosity"
    dt = η0 / G0 / 6.0 / 2  # assumes Maxwell time of 4
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

    # Simple shear: periodic in x, driven by the two y boundaries sliding past each other. Those
    # two faces carry no condition at all, which is how `VelocityBoundaryConditions` lets the
    # caller prescribe a velocity by hand: `flow_bcs!` leaves such faces untouched and the solver
    # only writes the interior, so the profile stored on them below survives every iteration.
    # They are built before the containers because `StokesArrays` sizes the momentum residuals
    # from them: a periodic direction needs one extra row for the seam face.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = false),
        periodic = (left = true, right = true, top = false, bot = false),
    )

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni, flow_bcs)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Simple shear: Vx varies linearly with y. `yVx` includes the two ghost rows that sit half a
    # cell outside the box, so evaluating the exact linear profile on them puts the plate velocity
    # ∓εbg·ly right on the walls (averaging a linear function is exact).
    yVx = grid.xi_vel[1][2]
    stokes.V.Vx .= PTArray(backend)([2 * (y - ly / 2) * εbg for _ in xvi[1], y in yVx])
    fill!(stokes.V.Vy, 0.0)
    # Wipe the interior so the solver has to reconstruct it. Only the y-range is trimmed: the two
    # y-boundary rows carry no boundary condition and so keep the prescribed plate velocity, while
    # `Vx[1, :]` and `Vx[end, :]` are the two faces of the periodic seam, which the solver owns.
    @views stokes.V.Vx[:, 2:(end - 1)] .= 0.0e0
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

    for _ in 1:nsteps

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

        push!(τII, maximum(stokes.τ.xy))
        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        # visualisation. The titles name what is actually drawn: the effective
        # viscoelastoplastic viscosity and the accumulated plastic strain rate localise the shear
        # band, and the bottom-right panel checks τxy against the analytic viscoelastic buildup.
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\log_{10}(\eta_{vep})", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = "convergence", titlesize = 35)
        ax3 = Axis(fig[1, 3], aspect = 1, title = L"\log_{10}(\dot{\varepsilon}^{pl}_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 3], aspect = 1, title = L"\tau_{xy}", titlesize = 35)
        h11 = heatmap!(ax1, xci..., Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
        lines!(ax2, iters.err_evo_it / nx, log10.(iters.err_evo_V), linewidth = 3, label = "V")
        lines!(ax2, iters.err_evo_it / nx, log10.(iters.err_evo_P), linewidth = 3, label = "P")
        ε_pl_floor = eps(eltype(stokes.ε_pl.II))
        h22 = heatmap!(ax3, xci..., Array(log10.(max.(stokes.ε_pl.II, ε_pl_floor))), colormap = :batlow)
        lines!(ax4, ttot, τII, color = :black, label = "numerical")
        lines!(ax4, ttot, sol, color = :red, label = "viscoelastic")
        Colorbar(fig[1, 2], h11)
        axislegend(ax2)
        axislegend(ax4; position = :rb)
        Colorbar(fig[2, 4], h22)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

n = 64
nx = n
ny = n
figdir = "ShearBands2D_DYREL_SimpleShearPeriodic"
# NOTE: the global grid is deliberately *not* declared periodic. The x-periodicity of this model
# is carried entirely by the velocity boundary conditions, which give the seam face its own
# momentum row. Passing `periodx` to `init_global_grid` instead makes ImplicitGlobalGrid report a
# global grid shrunk by the halo overlap (`nx_g() == nx - 2`), which is what `velocity_dofs` and
# `pressure_dof` normalise the residual norms by - so the reported convergence is measured against
# the wrong number of unknowns. This miniapp runs on a single rank.
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
@time main(igg; figdir = figdir, nx = nx, ny = ny)
