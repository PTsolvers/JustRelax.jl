# Mode-I (tensile) yielding with the Drucker-Prager tensile cap.
#
# Restrained uniaxial extension: the box is stretched horizontally while its vertical
# extent is held fixed, so the imposed volumetric strain rate drives the pressure into
# tension until the cap opens the material. This is the simplest setup in which the
# cap, rather than the shear cone, controls the solution, and it is meant as the
# smoke test for tensile behaviour: the pressure must stop at the tensile strength,
# the overstress above it must follow the Perzyna regularization, and the volumetric
# plastic strain must accumulate while the deviatoric stress stays far below the cone.
#
# What this model does NOT show: localized mode-I zones. The imposed boundary
# conditions prescribe the same volumetric strain rate everywhere, so the opening is
# domain-wide and the `opened fraction` column stays near one however the mesh, the
# regularization or the seeding is changed. Reproducing the localized tensile zones of
# Popov, Berlie and Kaus (2025, Fig. 6a, b and Fig. 9b) needs their configuration: a
# free surface, gravity and a depth-dependent strength, so that failed zones can open
# while their surroundings unload elastically.

const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Pkg;
Pkg.activate("miniapps");

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

using GeoParams, CairoMakie
using Random: MersenneTwister
import JustPIC.GridGeometryUtils as GGU

# HELPER FUNCTIONS ---------------------------------------------------------------

function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function _init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        if GGU.inside(GGU.Point(x, y), circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
        end
        return nothing
    end

    @parallel (@idx ni) _init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) _init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
end

# Fraction of cells that opened by more than a tenth of the maximum. Near one means
# the whole domain yielded in tension; a localized zone would give a small fraction.
function opened_fraction(EVol)
    E = Array(EVol)
    Emax = maximum(E)
    Emax ≤ 0 && return 0.0, 0.0
    return count(≥(0.1 * Emax), E) / length(E), Emax
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(
        igg; nx = 64, ny = 64, nsteps = 20, dt_factor = 1.0, η_reg = 1.0e-3,
        EII_seed = 0.2, seed = 1234, figdir = nothing,
    )

    # Physical domain ------------------------------------
    ly = 1.0e0
    lx = ly
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xci, xvi) = grid

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress (cohesion: C*cos(ϕ))
    ϕ = 30              # friction angle
    ψ = 0               # no dilation: the pressure drop is imposed, not self-induced
    C = τ_y
    η0 = 1.0
    G0 = 1.0
    Gi = G0 / 2.0       # softer inclusion
    εbg = 1.0           # horizontal extension rate; the vertical extent is restrained
    pT = -0.5           # tensile strength of the cap
    dt = dt_factor * η0 / G0 / 8.0

    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)

    # The inclusion is weaker in tension, so it reaches the cap first.
    soft_C = LinearSoftening((C / cosd(ϕ) / 2, C / cosd(ϕ)), (0.0, 1.0))
    pl_bg = DruckerPragerCap(; C = C / cosd(ϕ), ϕ = ϕ, η_vp = η_reg, Ψ = ψ, pT = pT, softening_C = soft_C)
    pl_inc = DruckerPragerCap(; C = C / cosd(ϕ) / 2, ϕ = ϕ, η_vp = η_reg, Ψ = ψ, pT = pT / 2, softening_C = soft_C)

    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl_bg)),
            Elasticity = el_bg,
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl_inc)),
            Elasticity = el_inc,
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1        # resolved by several cells even at nx = 32
    circle = GGU.Circle((0.5, 0.5), radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2.1)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # Popov et al. seed failure with random perturbations of the accumulated
    # deviatoric viscoplastic strain, which cohesion softening turns into weak cells.
    stokes.EII_pl .= PTArray(backend)(EII_seed .* rand(MersenneTwister(seed), ni...))

    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # restrained extension: Vx = x * εbg, Vy = 0
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([(x - 0.5lx) * εbg for x in xvi[1], _ in 1:(ny + 2)])
    fill!(stokes.V.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    t = 0.0
    local iters
    for it in 1:nsteps
        iters = solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, igg;
            kwargs = (verbose = false, iterMax = 50.0e3, nout = 1.0e3, viscosity_cutoff = (-Inf, Inf))
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)
        t += dt
        println("  it = $it, t = $t, Pmin = $(minimum(Array(stokes.P)))")
    end

    fraction, EVol_max = opened_fraction(stokes.EVol_pl)

    if figdir !== nothing
        take(figdir)
        fig = Figure(size = (1400, 500))
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"E_{Vol}^{pl}")
        ax2 = Axis(fig[1, 2], aspect = 1, title = "Pressure")
        ax3 = Axis(fig[1, 3], title = L"P \text{ vs } \tau_{II}")
        heatmap!(ax1, xci..., Array(stokes.EVol_pl), colormap = :batlow)
        heatmap!(ax2, xci..., Array(stokes.P), colormap = :vik, colorrange = (-1.0, 1.0))
        scatter!(ax3, vec(Array(stokes.P)), vec(Array(stokes.τ.II)); color = (:blue, 0.4), markersize = 3)
        vlines!(ax3, pT; color = :red, linewidth = 2)
        save(joinpath(figdir, "tensile_$(nx)x$(ny)_dt$(dt_factor)_etavp$(η_reg).png"), fig)
    end

    return (;
        nx, ny, dt, η_reg, pT,
        converged = iters.norm_Rx[end] < 1.0e-5 && iters.norm_Ry[end] < 1.0e-5 && iters.norm_∇V[end] < 1.0e-5,
        Pmin = minimum(Array(stokes.P)),
        overstress = pT - minimum(Array(stokes.P)),
        τII_max = maximum(Array(stokes.τ.II)),
        EVol_max,
        fraction,
    )
end

# REGULARIZATION, RESOLUTION AND TIMESTEP STUDY -----------------------------------
# What to read in the table:
#   - `converged` must be true for every row; the cap return map has no fallback that
#     silently accepts an unconverged state, so a false row is a real failure.
#   - `Pmin` must sit at or just beyond the tensile strength pT = -0.5, never far past it.
#   - `overstress` is how far Pmin sits below pT. It is the Perzyna overstress, not an
#     error, and it must grow with η_vp: at a negligible η_vp the pressure stops just
#     short of pT, at η_vp = 0.1 it goes past it. Its timestep dependence is a
#     diagnostic to report, not a monotone expectation.
#   - `EVol_max` must be positive: the cap is opening the material.
#   - `fraction` is near one by construction here (see the header).
#
# The step count is scaled with the timestep so that every row covers the same physical
# time; comparing rows run to different end times says nothing about the timestep.
function study(;
        resolutions = (32, 64, 128), dt_factors = (1.0, 0.5), nsteps = 20,
        η_regs = (1.0e-3, 1.0e-1), figdir = "TensileLocalization2D",
    )
    results = NamedTuple[]
    for η_reg in η_regs, n in resolutions, f in dt_factors
        igg = IGG(init_global_grid(n, n, 1; init_MPI = !(JustRelax.MPI.Initialized()))...)
        println("η_vp = $η_reg, resolution $n x $n, dt factor $f")
        push!(
            results, main(
                igg; nx = n, ny = n, nsteps = round(Int, nsteps / f),
                dt_factor = f, η_reg = η_reg, figdir = figdir
            )
        )
        # each resolution needs its own global grid, but MPI stays up for the next one
        finalize_global_grid(; finalize_MPI = false)
    end

    println("\n  nx  dt       eta_vp   converged  Pmin       overstress  tauII_max  EVol_max   opened fraction")
    for r in results
        println(
            lpad(r.nx, 4), "  ", rpad(round(r.dt; digits = 4), 8), " ",
            rpad(r.η_reg, 8), " ", rpad(r.converged, 10), " ",
            rpad(round(r.Pmin; digits = 4), 10), " ", rpad(round(r.overstress; digits = 4), 11), " ",
            rpad(round(r.τII_max; digits = 4), 10), " ", rpad(round(r.EVol_max; digits = 4), 10), " ",
            round(r.fraction; digits = 3),
        )
    end
    return results
end

study()
