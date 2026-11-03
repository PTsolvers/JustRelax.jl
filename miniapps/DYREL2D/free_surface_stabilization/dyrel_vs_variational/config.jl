# Knobs for the DYREL vs. variational-DYREL iteration comparison.
#
# Everything that defines the *model* is shared by both runs, so the two solvers are handed the
# same problem. Everything that is specific to one solver is listed under "per-solver" below,
# with the value the corresponding miniapp uses.

const SECYR = 3600 * 24 * 365.25

# ---- shared model ----------------------------------------------------------------------
const IS_CUDA = parse(Bool, get(ENV, "DYREL_CUDA", "false"))  # both runs share this backend
const N = parse(Int, get(ENV, "DYREL_N", "64"))  # cells per direction
const NSTEPS = 50                # time steps to compare
const THICK_AIR = 100.0e3        # sticky-air layer thickness [m]
const LX = 500.0e3               # domain width [m]
const LY = 500.0e3 + THICK_AIR   # domain height [m]
const AMPLITUDE = 5.0e3          # perturbation amplitude of the interface [m]
const SEED = 1234                # particles are seeded with `rand()`; fix it so runs match
const NXCELL = (125, 250, 75)    # nxcell, max_xcell, min_xcell (particles)
const NXCELL_CHAIN = (100, 75, 150)  # nxcell, min_xcell, max_xcell (marker chain)
const INIT_ELEVATION = -100.0e3  # marker-chain elevation [m]
const AIR_PHASE = 1

# A time step held constant for the whole run and shared by both solvers. `compute_dt` is
# velocity-dependent, and the two solvers produce different velocity fields, so an adaptive
# step would hand each solver a different problem and make "iterations per step"
# incomparable. `DYREL_DT_KYR` sets the step in kyr. Setting ADAPTIVE_DT = true recovers the
# miniapps' own behavior and gives up that comparability.
const ADAPTIVE_DT = false
const DT_KYR = parse(Float64, get(ENV, "DYREL_DT_KYR", "10.0"))
const DT = DT_KYR * 1.0e3 * SECYR
const DT_MAX = 50.0e3 * SECYR

# ---- shared solver settings ------------------------------------------------------------
# Every solver setting below is the one `RayleighTaylor2D_VariationalStokes_DYREL.jl` runs with,
# applied to both solvers so that a difference in iteration count is a difference between the
# solvers rather than between two tunings. A step that exhausts `total_iterMax` is recorded as
# unconverged rather than being silently treated as a cheap solve.
const DYREL_TOL = 1.0e-6
const C_FACT = 0.9               # damping scale of the dynamic-relaxation coefficients

# Dynamic-relaxation CFL, shared by both solvers; the default matches the package default.
# At 128² the standard solver's velocity solve diverges intermittently on this problem for steps
# of 25 kyr and above — geometrically, inside a single dynamic-relaxation sweep, in a mode
# localized in the sticky air just above the density interface. Lowering the CFL changes which
# runs survive but does not remove it, so a 50 kyr step has no reproducible result at 128².
const DR_CFL = parse(Float64, get(ENV, "DYREL_CFL", "0.99"))
const TOTAL_ITERMAX = parse(Float64, get(ENV, "DYREL_TOTAL_ITERMAX", "250000.0"))
const VERBOSE_PH = parse(Bool, get(ENV, "DYREL_VERBOSE", "false"))

# The solver's free-surface flag controls two separate things, and `DYREL_FREE_SURFACE=false`
# turns off both: the Kaus FSSA term `Vy·∂(ρg)/∂y·dt` in the vertical momentum residual, and —
# because the solver then hands `nothing` instead of `ρg` to `DYREL!` — the `fssa_penalty_floor`
# that otherwise puts a lower bound of `|Δ(ρg)|·dt·dy/2` on `γ_eff`. That floor is the only part
# of the penalty carrying an explicit grid spacing, so it is the suspect for a γ optimum that
# moves with resolution; switching this off is the diagnostic for it.
const FREE_SURFACE = parse(Bool, get(ENV, "DYREL_FREE_SURFACE", "true"))

const COMMON_SOLVER_KWARGS = (;
    total_iterMax = TOTAL_ITERMAX,
    nout = 100,
    rel_drop = 0.1,
    λ_relaxation_PH = 1,
    λ_relaxation_DR = 1,
    viscosity_relaxation = 1.0,
    linear_viscosity = true,
    free_surface = FREE_SURFACE,
    verbose_PH = VERBOSE_PH,
    verbose_DR = false,
    viscosity_cutoff = (-Inf, Inf),
)

# The standard solver takes a single `iterMax` for its velocity solve, matching `iterMax_DR`.
const SOLVER_KWARGS = (;
    COMMON_SOLVER_KWARGS...,
    iterMax = 50.0e3,
)

const VARIATIONAL_SOLVER_KWARGS = (;
    COMMON_SOLVER_KWARGS...,
    iterMax_PH = 50.0e3,
    iterMax_DR = 50.0e3,
)

# ---- per-solver ------------------------------------------------------------------------
# Penalty scaling of the Powell-Hestenes outer loop, and the dominant control on the iteration
# count. Both solvers take the same value so the comparison is like-for-like; `DYREL_GAMMA`
# overrides it.
const GAMMA_FACT_STANDARD = parse(Float64, get(ENV, "DYREL_GAMMA", "120.0"))
const GAMMA_FACT_VARIATIONAL = parse(Float64, get(ENV, "DYREL_GAMMA", "120.0"))

# `VelocityBoundaryConditions(; free_surface)`: the standard run stabilizes the sticky-air
# top through the boundary condition, the variational run through the marker chain + RockRatio.
const BC_FREE_SURFACE_STANDARD = true
const BC_FREE_SURFACE_VARIATIONAL = false

# ---- output ----------------------------------------------------------------------------
const OUTDIR = joinpath(@__DIR__, get(ENV, "DYREL_OUTDIR", "results"))
