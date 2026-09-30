# =========================================================================================
# ReservoirFailure2D.jl
#
# Injection-driven failure of a magma reservoir in a visco-elasto-plastic crust (2D,
# plane strain), with a free surface (variational Stokes + DYREL).
#
# Derived from `Arne_VS_refined-2.jl`. Changes with respect to that script:
#
#   1. Units of the source. In 2D, the injection rate is an AREA per time (m^2 per m
#      along strike), `Q2D` in km^2/yr. The old script non-dimensionalised a km^3/yr rate
#      and divided it by a 2D area, so the applied strain was Q*dt/(A*L_c) and changed
#      with the characteristic length (~7e-5 km^2/yr for 1e-3 km^3/yr and L_c = 14 km).
#   2. Magma properties are magma-like and configurable: bulk modulus K (sets the
#      overpressure per injected area), shear modulus G, viscosity and density.
#      The Poisson ratio passed to the elasticity is derived from K and G, so the
#      compressibility used by the solver and by the density law are the same.
#   3. Loading is a TIME schedule (inflate / withdraw / rest / re-inflate), not an
#      iteration schedule, because the time step is now adaptive.
#   4. Failure-aware time stepping: dt is reduced (down to `dt_min`) as the rock around
#      the chamber approaches the yield surface, so failure onset is resolved in time.
#      Steps never straddle a change of the loading schedule.
#   5. Failure diagnostics every step, written to `timeseries.csv`:
#        chamber overpressure, injected area, surface uplift, yield proximity,
#        and the time, place and mode (shear/tensile) of the first plastic yielding in
#        the rock around the chamber, recorded separately from the fault.
#   6. Clean-ups: the feeding pipe (which painted crust onto crust) is removed, the
#      temperature anomaly is set in one correct kernel, and the fault and background
#      strain rate are switchable, so the reference case isolates the reservoir.
#
# Failure uses the prescribed effective pressure P - α_B Pf in the solver and diagnostics.
# The undrained increment remains diagnostic-only. "tensile" denotes the rounded cap
# branch, not a propagated fracture; volumetric plasticity also detects pure opening.
# =========================================================================================

using Pkg;
Pkg.activate(".")

const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

import JustPIC.GridGeometryUtils as GGU

using GeoParams, Printf

Pkg.activate("miniapps")

using CairoMakie, PoissonGrids

# -----------------------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------------------
const ROCK_PHASE = 1
const MAGMA_PHASE = 2
const FAULT_PHASE = 3
const AIR_PHASE = 4

function init_phases!(
        phases, particles, xc_anomaly, yc_anomaly, r_anomaly,
        has_fault, xs_fault, α_fault, r_fault, maxd_fault, top, bottom,
    )
    ni = size(phases)

    @parallel_indices (i, j) function _init_phases!(
            phases, px, py, index, xc_anomaly, yc_anomaly, r_anomaly,
            has_fault, xs_fault, α_fault, r_fault, maxd_fault, top, bottom,
        )
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])   # positive downwards
            y = @index py[ip, i, j]

            if top ≤ depth ≤ bottom
                @index phases[ip, i, j] = Float64(ROCK_PHASE)
            end

            # pre-existing weak fault (optional)
            if has_fault && (0 ≤ depth ≤ maxd_fault)
                xc_fault = xs_fault + depth / tand(α_fault)
                if abs(x - xc_fault) ≤ r_fault
                    @index phases[ip, i, j] = Float64(FAULT_PHASE)
                end
            end

            # magma reservoir (circular in 2D)
            if (x - xc_anomaly)^2 + (y - yc_anomaly)^2 ≤ r_anomaly^2
                @index phases[ip, i, j] = Float64(MAGMA_PHASE)
            end

            if depth < top
                @index phases[ip, i, j] = Float64(AIR_PHASE)
            end
        end
        return nothing
    end

    return @parallel (@idx ni) _init_phases!(
        phases, particles.coords..., particles.index, xc_anomaly, yc_anomaly, r_anomaly,
        has_fault, xs_fault, α_fault, r_fault, maxd_fault, top, bottom,
    )
end

# Circular temperature anomaly on the cell centres (thermal.T carries one ghost layer).
@parallel_indices (i, j) function _circular_anomaly!(T, Tanomaly, xc, yc, r, x, y)
    if (x[i] - xc)^2 + (y[j] - yc)^2 ≤ r^2
        T[i + 1, j + 1] = Tanomaly
    end
    return nothing
end

function circular_anomaly!(T, Tanomaly, xc, yc, r, xci)
    ni = size(T) .- 2
    @parallel (@idx ni) _circular_anomaly!(T, Tanomaly, xc, yc, r, xci...)
    return nothing
end

@parallel_indices (i, j) function set_air_particle_temperature!(pT, pPhases, index, air_phase, Ttop)
    for ip in cellaxes(index)
        if @index(index[ip, i, j]) && @index(pPhases[ip, i, j]) == air_phase
            @index pT[ip, i, j] = Ttop
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------------------
# Rheology
# -----------------------------------------------------------------------------------------
function creep_models(; linear = false, η_magma = 1.0e16Pa * s)
    creep_magma = LinearViscous(; η = η_magma)
    creep_air = LinearViscous(; η = 1.0e18Pa * s)
    if linear
        creep_rock = LinearViscous(; η = 1.0e23Pa * s)
        creep_fault = LinearViscous(; η = 1.0e20Pa * s)
    else
        creep_rock = DislocationCreep(;
            n = 3.3NoUnits,
            A = 1.0 * exp10(-5.7)MPa^(-33 // 10) / s,
            E = 186.5kJ / mol,
            V = 0m^3 / mol,
            r = 0NoUnits,
            Apparatus = AxialCompression,
        )
        creep_fault = DislocationCreep(;
            A = 1.67e-22Pa^(-(35 // 10)) / s,
            n = 3.5,
            E = 1.87e5J / mol,
            V = 0 * 6.0e-6m^3 / mol,
            r = 0.0,
            R = 8.3145J / mol / K,
        )
    end
    return creep_rock, creep_magma, creep_fault, creep_air
end

# Poisson ratio consistent with a given (K, G)
poisson_ratio(K, G) = (3K - 2G) / (2 * (3K + G))

function init_rheology(creeps, p, CD; is_compressible = true)
    creep_rock, creep_magma, creep_fault, creep_air = creeps

    η_reg = 1.0e19Pa * s
    Coh = p.C
    ϕ = p.ϕ
    Ψ = p.Ψ

    0 < p.C_residual_fraction ≤ 1 || throw(ArgumentError("C_residual_fraction must be in (0, 1]"))
    0 ≤ p.C_softening_start < p.C_softening_end || throw(ArgumentError("cohesion softening requires 0 ≤ start < end"))
    soft_C = LinearSoftening(Coh * p.C_residual_fraction, Coh, p.C_softening_start, p.C_softening_end)
    pl_c = DruckerPragerCap(; C = Coh, ϕ = ϕ, η_vp = η_reg, Ψ = Ψ, pT = p.pT, softening_C = soft_C)
    pl_fc = DruckerPragerCap(; C = Coh / 2, ϕ = ϕ / 2, η_vp = η_reg, Ψ = Ψ, pT = p.pT / 4)

    G_rock = p.G_rock
    K_rock = p.K_rock
    G_magma = p.G_magma
    K_magma = p.K_magma

    if is_compressible
        el = SetConstantElasticity(; G = G_rock, ν = poisson_ratio(K_rock, G_rock))
        el_magma = SetConstantElasticity(; G = G_magma, ν = poisson_ratio(K_magma, G_magma))
        β_rock = inv(ustrip(uconvert(Pa, K_rock)))
        β_magma = inv(ustrip(uconvert(Pa, K_magma)))
    else
        el = SetConstantElasticity(; G = G_rock, ν = 0.5)
        el_magma = SetConstantElasticity(; G = G_magma, ν = 0.5)
        β_rock = inv(get_Kb(el) * GPa)
        β_magma = inv(get_Kb(el_magma) * GPa)
    end

    g = 9.81m / s^2
    return (
        # Crust
        SetMaterialParams(;
            Phase = ROCK_PHASE,
            Density = PT_Density(; ρ0 = p.ρ_rock, α = 3.0e-5 / K, T0 = 0.0C, β = β_rock / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_rock, el, pl_c)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CD,
        ),
        # Magma
        SetMaterialParams(;
            Phase = MAGMA_PHASE,
            Density = PT_Density(; ρ0 = p.ρ_magma, α = 3.0e-5 / K, T0 = 0.0C, β = β_magma / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 1.5Watt / K / m),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_magma, el_magma)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_magma,
            CharDim = CD,
        ),
        # Fault
        SetMaterialParams(;
            Phase = FAULT_PHASE,
            Density = PT_Density(; ρ0 = p.ρ_rock, α = 3.0e-5 / K, T0 = 0.0C, β = β_rock / Pa),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = 1.0e-6Watt / m^3),
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((creep_fault, el, pl_fc)),
            Melting = MeltingParam_Caricchi(),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el,
            CharDim = CD,
        ),
        # Sticky air
        SetMaterialParams(;
            Phase = AIR_PHASE,
            Density = ConstantDensity(ρ = 1kg / m^3),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1000 * 1.0e3J / kg / K),
            Conductivity = ConstantConductivity(; k = 15Watt / K / m),
            LatentHeat = ConstantLatentHeat(; Q_L = 0.0J / kg),
            ShearHeat = ConstantShearheating(0.0NoUnits),
            CompositeRheology = CompositeRheology((creep_air, el)),
            Gravity = ConstantGravity(; g = g),
            CharDim = CD,
        ),
    )
end

# -----------------------------------------------------------------------------------------
# Default physical parameters (dimensional). Override any of them through `main(...; kw...)`.
# -----------------------------------------------------------------------------------------
default_params() = (;
    # geometry
    D = 30.0km,               # depth of the domain
    sticky_air = 5.0km,
    Lx = 50.0km,
    chamber_depth = 5.0km,
    chamber_radius = 1.5km,
    source_fraction = 0.5,    # injecting core radius / chamber radius
    # thermal
    T_top = 20C,
    T_bot = 438.5625C,
    T_magma = 1300C,          # imposed on magma particles during pre-heating
    t_preheat = 50.0e3yr,     # duration of the thermal pre-conditioning (chamber "age")
    n_preheat = 50,
    # rock
    ρ_rock = 2650kg / m^3,
    G_rock = 30GPa,
    K_rock = 50GPa,
    C = 20.0MPa,
    C_residual_fraction = 0.1, # linear cohesion decrease to C/2
    C_softening_start = 0.0,  # accumulated equivalent plastic strain
    C_softening_end = 0.1,
    ϕ = 30.0,
    Ψ = 10.0,
    pT = -10.0MPa,
    # magma
    ρ_magma = 2500kg / m^3,
    G_magma = 1.0GPa,
    K_magma = 5.0GPa,         # volatile-bearing magma: ~1-10 GPa
    η_magma = 1.0e16Pa * s,
    # tectonics and weaknesses
    εbg = 0.0 / s,            # background extension (e.g. 1e-15/s)
    has_fault = false,
    α_fault = 60.0,
    L_fault = 2.0km,
    r_fault = 0.25km,
    # loading: 2D injection rate (area per time, per metre along strike)
    Q2D = 1.0e-4km^2 / yr,
    # schedule: (end time of stage, multiplier of Q2D); last stage runs to t_end
    schedule = ((3.0e3yr, 1.0), (5.0e3yr, -1.0), (6.0e3yr, 0.0), (1.2e4yr, 1.0)),
    t_end = 1.2e4yr,
    # time stepping
    dt_ini = 10.0yr,
    dt_max = 100.0yr,
    dt_min = 1.0 / 365.25 * yr,   # one day
    yield_threshold = 0.8,         # start shrinking dt above this yield proximity
    dt_safety = 0.25,
    # diagnostics
    λ_mode = :thermal,
    λ_pf = 1.0,                    # uniform pore-pressure ratio when λ_mode = :uniform
    λ_cold = 0.38,
    λ_hot = 0.95,
    T1 = 350C,
    T2 = 450C,
    α_B = 1.0,
    B = 0.7,
    Λ = 0.1MPa / K,
    λ_fault = nothing,
    halo_width = 3.0km,            # rock within this distance of the chamber wall is tracked
    εpl_threshold = 1.0e-16 / s,   # plastic strain rate that counts as yielding
    # IO
    figdir = "ReservoirFailure2D",
    nout = 5,
)

merge_params(; kw...) = merge(default_params(), (; kw...))

# -----------------------------------------------------------------------------------------
# Injection
# -----------------------------------------------------------------------------------------
# `Q` holds the volumetric strain each cell undergoes over one time step: `q` in the
# cells flagged by `ind`, zero elsewhere. `buffer` is host scratch of `size(Q)`.
function set_volumetric_source!(Q, buffer, ind, q)
    buffer .= 0.0
    buffer[ind] .= q
    copyto!(Q, buffer)
    return Q
end

# Stage multiplier and the time at which the current stage ends.
function loading_stage(t, schedule_nd)
    for (t_stage_end, mult) in schedule_nd
        t < t_stage_end && return mult, t_stage_end
    end
    return last(schedule_nd)[2], Inf
end

# -----------------------------------------------------------------------------------------
# Variational pressure helpers (see Arne_VS_refined-2.jl)
# -----------------------------------------------------------------------------------------
@parallel_indices (i, j) function rock_buoyancy!(ρg_rock, ρg, rock_fraction)
    ρg_rock[i, j] = ρg[i, j] * rock_fraction[i, j]
    return nothing
end

@parallel_indices (i, j) function _rescale_variational_pressure!(P, rock_fraction)
    ϕ = rock_fraction[i, j]
    P[i, j] = ϕ > 0 ? P[i, j] / ϕ : zero(eltype(P))
    return nothing
end

rescale_variational_pressure!(P, ϕ::JustRelax.RockRatio) =
    @parallel (@idx size(P)) _rescale_variational_pressure!(P, ϕ.center)

function lithostatic_pressure!(Plitho, ρg_litho, ρg, ϕ_R, grid, igg)
    ni = size(Plitho)
    @parallel (@idx ni) rock_buoyancy!(ρg_litho, ρg, ϕ_R.center)
    compute_lithostatic_pressure!(Plitho, ρg_litho, grid.di.vertex[2], igg)
    rescale_variational_pressure!(Plitho, ϕ_R)
    return Plitho
end

# -----------------------------------------------------------------------------------------
# Failure diagnostics (host side; pressure and temperature converted to physical units)
# -----------------------------------------------------------------------------------------
# Normalized exact yield margin: r = 1 on the cap/cone, r < 1 inside.
# The stress scale is fixed, avoiding singular ratios near zero shear strength.
@inline function yield_proximity(τII, P, cap, EII = 0.0)
    (; scale, model) = cap
    C_eff = model.softening_C(EII, model.C.val)
    cp = GeoParams.compute_tensile_cap(model.sinϕ.val, model.cosϕ.val, model.sinΨ.val, C_eff, model.pT.val)
    F = GeoParams.compute_yieldfunction(model; P, τII, EII)
    tensile = τII * (cp.py - cp.pd) < cp.τd * (cp.py - P)
    return max(0.0, 1 + F / scale), tensile
end

@inline plastic_activity(εII, εvol, λ) = max(εII, abs(εvol), λ)

@inline function pore_pressure_ratio(T, p)
    λ = if p.λ_mode == :uniform
        p.λ_pf
    else
        ξ = (T - p.T1_C) / (p.T2_C - p.T1_C)
        ξ = clamp(ξ, 0.0, 1.0)
        p.λ_cold + (p.λ_hot - p.λ_cold) * ξ^2 * (3 - 2ξ)
    end
    return λ
end

# Build once per mechanical step; diagnostics receive these same fields.
function prescribed_pore_pressure(T, Plitho, phases, rock_fraction, p)
    λ = zeros(size(Plitho))
    for I in CartesianIndices(λ)
        phase = argmax(phases[I])
        if rock_fraction[I] > 0 && (phase == ROCK_PHASE || phase == FAULT_PHASE)
            λ[I] = phase == FAULT_PHASE && p.λ_fault !== nothing ? p.λ_fault : pore_pressure_ratio(T[I], p)
        end
    end
    return λ .* Plitho, λ
end

"""
    failure_diagnostics(...)

Returns chamber overpressure, total/effective/undrained yield proximity in the halo,
yield locations and status, and cell-centred pore-pressure diagnostic fields.
`x_rmax`, `y_rmax` and `tensile_at_max` refer to the effective-stress maximum.
The pore-pressure fields are those used for this mechanical step, before updating lithostatic pressure.
"""
function failure_diagnostics(
        stokes, thermal, Plitho, P_start, T_start, phase_ratios, ϕ_R, xci_cpu, cap, p_nd, CD; chamber, Pf_used, λ_used,
    )
    P = ustrip.(dimensionalize(Array(stokes.P), MPa, CD))
    Pl = ustrip.(dimensionalize(Array(Plitho), MPa, CD))
    τII = ustrip.(dimensionalize(Array(stokes.τ.II), MPa, CD))
    T = ustrip.(dimensionalize(Array(thermal[2:(end - 1), 2:(end - 1)]), C, CD))
    Δσm = P .- P_start
    ΔT = T .- T_start
    EII = Array(stokes.EII_pl)
    εpl = plastic_activity.(Array(stokes.ε_pl.II), Array(stokes.ε_vol_pl), Array(stokes.λ))
    ph = [argmax(p) for p in Array(phase_ratios.center)]
    fR = Array(ϕ_R.center)

    (; xc, yc, r) = chamber
    Pf = ustrip.(dimensionalize(Array(Pf_used), MPa, CD))
    Δp_undrained = zeros(size(P))
    λ_field = λ_used

    ΔP_sum, n_magma = 0.0, 0
    r_max, r_max_eff, r_max_undrained = 0.0, 0.0, 0.0
    I_max = CartesianIndex(1, 1)
    tensile_at_max = false
    halo_yield, fault_yield = false, false
    halo_yield_I = CartesianIndex(1, 1)
    halo_yield_max_εpl = 0.0
    halo_yield_tensile = false

    for I in CartesianIndices(P)
        i, j = Tuple(I)
        if fR[I] > 0.0 && (ph[I] == ROCK_PHASE || ph[I] == FAULT_PHASE)
            Δp_undrained[I] = p_nd.B * Δσm[I] + p_nd.Λ_MPa_K * ΔT[I]
        end
        fR[I] < 0.5 && continue
        if ph[I] == MAGMA_PHASE
            ΔP_sum += P[I] - Pl[I]
            n_magma += 1
            continue
        end
        d = sqrt((xci_cpu[1][i] - xc)^2 + (xci_cpu[2][j] - yc)^2) - r
        if ph[I] == FAULT_PHASE
            fault_yield |= εpl[I] > p_nd.εpl_threshold
            continue
        end
        (ph[I] == ROCK_PHASE && 0 ≤ d ≤ p_nd.halo_width) || continue

        rr, _ = yield_proximity(τII[I], P[I], cap, EII[I])
        P_eff = P[I] - p_nd.α_B * Pf[I]
        P_eff_undrained = P[I] - p_nd.α_B * (Pf[I] + Δp_undrained[I])
        rr_eff, is_tens = yield_proximity(τII[I], P_eff, cap, EII[I])
        rr_undrained, _ = yield_proximity(τII[I], P_eff_undrained, cap, EII[I])
        r_max_undrained = max(r_max_undrained, rr_undrained)
        r_max = max(r_max, rr)
        if rr_eff > r_max_eff
            r_max_eff, I_max, tensile_at_max = rr_eff, I, is_tens
        end
        if εpl[I] > p_nd.εpl_threshold && εpl[I] > halo_yield_max_εpl
            halo_yield = true
            halo_yield_I = I
            halo_yield_max_εpl = εpl[I]
            halo_yield_tensile = is_tens
        end
    end

    i, j = Tuple(halo_yield_I)
    return (;
        ΔP_chamber = n_magma > 0 ? ΔP_sum / n_magma : 0.0,
        r_max,
        r_max_eff,
        x_rmax = xci_cpu[1][I_max[1]],
        y_rmax = xci_cpu[2][I_max[2]],
        tensile_at_max,
        halo_yield,
        fault_yield,
        x_yield = halo_yield ? xci_cpu[1][i] : NaN,
        y_yield = halo_yield ? xci_cpu[2][j] : NaN,
        lambda_yield = halo_yield ? λ_field[halo_yield_I] : NaN,
        Pf,
        Δp_undrained,
        λ_field,
        r_max_undrained,
        # classified by the stress state at the yielding cell (the dilatant DP flow rule
        # also produces positive volumetric plastic strain, so EVol_pl cannot tell them apart)
        yield_mode = halo_yield ? (halo_yield_tensile ? "tensile" : "shear") : "none",
    )
end

# Failure-aware time step: shrink dt when the halo approaches the yield surface,
# using the growth rate of the yield proximity over the last step.
function failure_limited_dt(dt_next, dt, r, r_old, p_nd)
    dt_next = min(dt_next, 2 * dt)          # limit growth after a refined phase
    # only refine on the approach to yield; once yielding (r ≥ 1) let dt recover
    (r < p_nd.yield_threshold || r ≥ 1) && return dt_next
    drdt = (r - r_old) / dt
    dt_fail = drdt > 0 ? p_nd.dt_safety * max(1 - r, 0.0) / drdt : dt_next
    return clamp(min(dt_next, dt_fail), p_nd.dt_min, dt_next)
end

# Report the dominant phase, matching VTK `phase`; a mixed cell has no single
# equivalent DYREL yield surface. Nonplastic phases have undefined strength (NaN).
function softened_strength_output(EII_pl, phases, rheology, CD)
    strain, ratios = Array(EII_pl), Array(phases)
    cohesion = fill(NaN, size(strain))
    friction = fill(NaN, size(strain))
    for I in CartesianIndices(strain)
        phase = argmax(ratios[I])
        is_pl, C, sinϕ, cosϕ, _, _ = JustRelax.JustRelax2D.plastic_params(rheology[phase], strain[I])
        if is_pl
            cohesion[I] = C
            friction[I] = atand(sinϕ, cosϕ)
        end
    end
    return (;
        cohesion_MPa = ustrip.(dimensionalize(cohesion, MPa, CD)),
        friction_angle_deg = friction,
    )
end

# Local source contribution for the latest step, not accumulated or transported heat.
# Evaluate heat capacity at the converged thermal state, before particle advection.
function shear_heating_output(thermal, P, phases, rheology, dt, CD)
    H = Array(thermal.shear_heating)
    T = Array(thermal.T[2:(end - 1), 2:(end - 1)])
    pressure, ratios = Array(P), Array(phases)
    ΔT = similar(H)
    for I in CartesianIndices(H)
        ρCp = JustRelax.JustRelax2D.compute_ρCp(rheology, ratios[I], (; T = T[I], P = pressure[I]))
        ΔT[I] = H[I] * dt / ρCp
    end
    return (;
        shear_heating_W_m3 = ustrip.(dimensionalize(H, J / s / m^3, CD)),
        dT_shear_source_K = ustrip.(dimensionalize(ΔT, K, CD)),
    )
end

# -----------------------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------------------
const TS_HEADER = "it,t_yr,dt_yr,Q_mult,A_injected_km2,dP_chamber_MPa,uplift_max_m," *
    "r_max,r_max_eff,r_max_undrained,lambda_at_yield,x_rmax_km,y_rmax_km,tensile_at_max,halo_yield,fault_yield," *
    "x_yield_km,y_yield_km,yield_mode"

function write_timeseries_row(io, it, t, dt, mult, A_inj, d, uplift, CD)
    tyr(x) = ustrip(dimensionalize(x, yr, CD))
    km_(x) = ustrip(dimensionalize(x, km, CD))
    MPa_(x) = ustrip(dimensionalize(x, MPa, CD))
    @printf(
        io, "%d,%.6e,%.6e,%.3f,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%d,%d,%.4f,%.4f,%s\n",
        it, tyr(t), tyr(dt), mult, ustrip(dimensionalize(A_inj, km^2, CD)),
        MPa_(d.ΔP_chamber), uplift, d.r_max, d.r_max_eff, d.r_max_undrained, d.lambda_yield,
        km_(d.x_rmax), km_(d.y_rmax), d.tensile_at_max, d.halo_yield, d.fault_yield,
        km_(d.x_yield), km_(d.y_yield), d.yield_mode,
    )
    return flush(io)
end

function plot_snapshot(figdir, it, t, stokes, thermal, Plitho, chain, xci_cpu, xvi_cpu, li, CD, onset)
    km_(x) = ustrip.(dimensionalize(x, km, CD))
    xc_km, yc_km = km_(xci_cpu[1]), km_(xci_cpu[2])
    chain_x = km_(Array(chain.coords[1].data[:]))
    chain_y = km_(Array(chain.coords[2].data[:]))
    t_kyr = ustrip(dimensionalize(t, yr, CD)) / 1.0e3

    fig = Figure(; size = (1600, 1200))
    ar = li[1] / li[2]
    Label(fig[0, 1:2], @sprintf("t = %.3f kyr", t_kyr); fontsize = 28)
    ax1 = Axis(fig[1, 1][1, 1]; aspect = ar, title = "T [°C]")
    ax2 = Axis(fig[1, 2][1, 1]; aspect = ar, title = "log10 η_vep [Pa s]")
    ax3 = Axis(fig[2, 1][1, 1]; aspect = ar, title = "P − P_litho [MPa]")
    ax4 = Axis(fig[2, 2][1, 1]; aspect = ar, title = "log10 ε̇II_pl [1/s]")

    h1 = heatmap!(ax1, xc_km, yc_km, ustrip.(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), C, CD)); colormap = :batlow)
    h2 = heatmap!(ax2, xc_km, yc_km, log10.(ustrip.(dimensionalize(Array(stokes.viscosity.η_vep), Pa * s, CD))); colormap = :glasgow, colorrange = (16, 24))
    ΔP = ustrip.(dimensionalize(Array(stokes.P .- Plitho), MPa, CD))
    lim = max(maximum(abs, ΔP), 1.0e-3)
    h3 = heatmap!(ax3, xc_km, yc_km, ΔP; colormap = :roma, colorrange = (-lim, lim))
    εpl = ustrip.(dimensionalize(Array(stokes.ε_pl.II), s^-1, CD))
    h4 = heatmap!(ax4, xc_km, yc_km, log10.(max.(εpl, 1.0e-22)); colormap = :lipari, colorrange = (-18, -12))
    for ax in (ax1, ax2, ax3, ax4)
        scatter!(ax, chain_x, chain_y; color = :red, markersize = 3)
    end
    if onset !== nothing
        scatter!(ax4, [km_(onset.x)], [km_(onset.y)]; color = :cyan, marker = :star5, markersize = 25)
    end
    Colorbar(fig[1, 1][1, 2], h1)
    Colorbar(fig[1, 2][1, 2], h2)
    Colorbar(fig[2, 1][1, 2], h3)
    Colorbar(fig[2, 2][1, 2], h4)
    linkaxes!(ax1, ax2, ax3, ax4)
    save(joinpath(figdir, @sprintf("snapshot_%06d.png", it)), fig)
    return nothing
end

function plot_timeseries(figdir, rows)
    isempty(rows) && return nothing
    t = [r.t for r in rows]
    fig = Figure(; size = (1000, 900))
    ax1 = Axis(fig[1, 1]; ylabel = "ΔP chamber [MPa]")
    ax2 = Axis(fig[2, 1]; ylabel = "max uplift [m]")
    ax3 = Axis(fig[3, 1]; ylabel = "yield proximity", xlabel = "t [yr]")
    lines!(ax1, t, [r.ΔP for r in rows])
    lines!(ax2, t, [r.uplift for r in rows])
    lines!(ax3, t, [r.r for r in rows]; label = "total P")
    lines!(ax3, t, [r.r_eff for r in rows]; label = "effective P", linestyle = :dash)
    lines!(ax3, t, [r.r_undrained for r in rows]; label = "undrained", linestyle = :dashdot)
    hlines!(ax3, [1.0]; color = :black, linestyle = :dot)
    axislegend(ax3; position = :lt)
    tons = [r.t for r in rows if r.halo_yield]
    if !isempty(tons)
        for ax in (ax1, ax2, ax3)
            vlines!(ax, [first(tons)]; color = :red, linestyle = :dash)
        end
    end
    linkxaxes!(ax1, ax2, ax3)
    save(joinpath(figdir, "timeseries.png"), fig)
    return nothing
end

# -----------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------
function main(igg, nx, ny; do_vtk = false, linear = false, kw...)
    p = merge_params(; kw...)

    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    nd(x) = nondimensionalize(x, CD)

    p.λ_mode in (:thermal, :uniform) || throw(ArgumentError("λ_mode must be :thermal or :uniform"))
    p.λ_mode == :uniform && !(0 ≤ p.λ_pf ≤ 1) && throw(ArgumentError("λ_pf must be between 0 and 1"))
    p.T2 > p.T1 || throw(ArgumentError("T2 must be greater than T1"))
    0 ≤ p.B ≤ 1 || throw(ArgumentError("B must be between 0 and 1"))
    0 ≤ p.α_B ≤ 1 || throw(ArgumentError("α_B must be between 0 and 1"))
    0 ≤ p.λ_cold ≤ 1 && 0 ≤ p.λ_hot ≤ 1 || throw(ArgumentError("λ_cold and λ_hot must be between 0 and 1"))
    p.λ_fault === nothing || 0 ≤ p.λ_fault ≤ 1 || throw(ArgumentError("λ_fault must be between 0 and 1"))

    # non-dimensional copies of the parameters used inside the time loop
    p_nd = (;
        dt_min = nd(p.dt_min),
        yield_threshold = p.yield_threshold,
        dt_safety = p.dt_safety,
        λ_pf = p.λ_pf,
        λ_mode = p.λ_mode,
        λ_cold = p.λ_cold,
        λ_hot = p.λ_hot,
        T1_C = ustrip(uconvert(C, p.T1)),
        T2_C = ustrip(uconvert(C, p.T2)),
        α_B = p.α_B,
        B = p.B,
        Λ_MPa_K = ustrip(uconvert(MPa / K, p.Λ)),
        λ_fault = p.λ_fault,
        halo_width = nd(p.halo_width),
        εpl_threshold = nd(p.εpl_threshold),
    )
    cap = (;
        model = DruckerPragerCap(;
            C = ustrip(uconvert(MPa, p.C)), ϕ = p.ϕ, Ψ = p.Ψ,
            pT = ustrip(uconvert(MPa, p.pT)),
            softening_C = LinearSoftening(
                ustrip(uconvert(MPa, p.C)) * p.C_residual_fraction, ustrip(uconvert(MPa, p.C)),
                p.C_softening_start, p.C_softening_end,
            ),
        ),
        cp = GeoParams.compute_tensile_cap(
            sind(p.ϕ), cosd(p.ϕ), sind(p.Ψ),
            ustrip(uconvert(MPa, p.C)), ustrip(uconvert(MPa, p.pT)),
        ),
        scale = abs(ustrip(uconvert(MPa, p.pT))),
    )
    p.pT < 0MPa || throw(ArgumentError("pT must be negative"))
    schedule_nd = Tuple((nd(te), m) for (te, m) in p.schedule)
    t_end = nd(p.t_end)

    # Domain --------------------------------------------------------------------------
    sticky_air = nd(p.sticky_air)
    D = nd(p.D)
    lx = nd(p.Lx)
    ly = D + sticky_air
    li = lx, ly
    ni = nx, ny
    origin = -lx / 2, -D
    M = window_monitor(5.0, 4.0, nd(2.0km), 0.0e0) # refinement around the chamber axis
    xv_ref = solve_grid(-lx / 2, lx / 2, M, nx)
    yv_ref = collect(LinRange(-D, sticky_air, ny + 1))
    grid = Geometry(PTArray(backend), xv_ref, yv_ref)
    (; xci, xvi) = grid
    xci_cpu = Array.(xci)
    xvi_cpu = Array.(xvi)
    di_min = minimum.(grid.di.vertex)
    εbg = nd(p.εbg)

    # Rheology ------------------------------------------------------------------------
    creeps = creep_models(; linear = linear, η_magma = p.η_magma)
    # Solver fields and time steps use CD units. Keep material parameters in the
    # same system; otherwise PT_Density receives nondimensional P/T with SI
    # coefficients and can return negative density/ρCp.
    rheology = nondimensionalize(
        init_rheology(creeps, p, CD; is_compressible = true),
        CD,
    )
    cutoff_visc = nd.((1.0e16Pa * s, 1.0e24Pa * s))
    dt = nd(p.dt_ini)
    dt_max = nd(p.dt_max)
    Q2D = nd(p.Q2D)                    # area / time (2D)

    # Particles -----------------------------------------------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 15
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, Array.(grid.xi_vel[1]), Array.(grid.xi_vel[2]))
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    pT, pPhases = init_cell_arrays(particles, Val(2))

    # Chamber and fault geometry ------------------------------------------------------
    x_ch = 0.0
    y_ch = -nd(p.chamber_depth)
    r_ch = nd(p.chamber_radius)
    r_src = p.source_fraction * r_ch
    chamber = (; xc = x_ch, yc = y_ch, r = r_ch)
    xs_fault = x_ch - r_ch
    maxd_fault = nd(p.L_fault) * sind(p.α_fault)

    init_phases!(
        pPhases, particles, x_ch, y_ch, r_ch,
        p.has_fault, xs_fault, p.α_fault, nd(p.r_fault), maxd_fault,
        nd(0.0km), nd(20km),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Marker chain (free surface) -----------------------------------------------------
    chain = init_markerchain(backend_JP, 50, 25, 100, xvi[1], 0.0e0)
    h0 = copy(Array(chain.h_vertices))
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Temperature ---------------------------------------------------------------------
    thermal = ThermalArrays(backend, ni)
    Ttop = nd(p.T_top)
    Tbot = nd(p.T_bot)
    T_magma = nd(p.T_magma)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
    )
    ∇Tz = (Ttop - Tbot) / D
    T1D = @. (∇Tz * xci[2] + Ttop) * (xci[2] < 0.0e0)
    T1D[xci[2] .≥ 0.0e0] .= Ttop
    thermal.T[:, 2:(end - 1)] .+= PTArray(backend)(T1D')
    circular_anomaly!(thermal.T, T_magma, x_ch, y_ch, r_ch, grid.xci)
    thermal_bcs!(thermal, thermal_bc)

    ϕ_R = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ_R, chain, xvi, grid.di.vertex)

    # Stokes --------------------------------------------------------------------------
    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    ρg_litho = @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P); air_phase = AIR_PHASE)
        lithostatic_pressure!(stokes.P, ρg_litho, ρg[2], ϕ_R, grid, igg)
    end
    Plitho = copy(stokes.P)
    @copy thermal.Told thermal.T
    stokes.ε.xx .= max(εbg, nd(1.0e-20 / s))
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc; air_phase = AIR_PHASE)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di_min, li; ϵ = 1.0e-5, CFL = 0.8 / √2.1
    )

    # Injecting core ------------------------------------------------------------------
    ind = zeros(Bool, nx, ny)
    for i in 1:nx, j in 1:ny
        ind[i, j] = (xci_cpu[1][i] - x_ch)^2 + (xci_cpu[2][j] - y_ch)^2 ≤ r_src^2
    end
    Q_buffer = zeros(ni...)
    dx_host, dy_host = Array.(grid.di.vertex)
    source_area = sum(dx_host[i] * dy_host[j] for i in axes(ind, 1), j in axes(ind, 2) if ind[i, j])
    source_area > 0 || error("the volumetric source mask contains no cells")
    # Q2D [area/time] * dt [time] / source_area [area] = volumetric strain per step (dimensionless)

    # Far-field boundary conditions ---------------------------------------------------
    stokes.V.Vx .= PTArray(backend)([εbg * x for x in xvi_cpu[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-εbg * (y - sticky_air) for _ in 1:(nx + 2), y in xvi_cpu[2]])
    flow_bcs = VelocityBoundaryConditions(;
        no_slip = (left = false, right = false, top = false, bot = true),
        free_slip = (left = true, right = true, top = true, bot = false),
        periodic = (left = false, right = false, top = false, bot = false),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------------------------------------
    figdir = p.figdir
    take(figdir)
    vtk_dir = joinpath(figdir, "vtk")
    do_vtk && take(vtk_dir)
    ts_io = open(joinpath(figdir, "timeseries.csv"), "w")
    println(ts_io, TS_HEADER)
    rows = NamedTuple[]

    # Thermal pre-conditioning (sets the "age" of the thermal halo) -------------------
    dt_pre = nd(p.t_preheat) / p.n_preheat
    centroid2particle!(pT, thermal.T, particles)
    for _ in 1:p.n_preheat
        heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, rheology, args, dt_pre, grid;
            kwargs = (; igg = igg, phase = phase_ratios, iterMax = 10.0e3, nout = 1.0e2, verbose = false),
        )
        centroid2particle!(pT, thermal.T, particles)
        pT.data[pPhases.data .== MAGMA_PHASE] .= T_magma
        particle2centroid!(thermal.T, pT, particles)
        thermal.Told .= thermal.T
    end

    dt₀ = similar(thermal.T)
    centroid2particle!(pT, thermal.T, particles)
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ_R, grid.di, dt; ϵ = 1.0e-3)

    # Time loop -----------------------------------------------------------------------
    t, it = 0.0, 0
    P_start = ustrip.(dimensionalize(Array(stokes.P), MPa, CD))
    T_start = ustrip.(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), C, CD))
    A_injected = 0.0
    r_old = 0.0
    onset = nothing
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    while t < t_end
        # loading stage; never step across a stage boundary
        mult, t_stage_end = loading_stage(t, schedule_nd)
        dt = min(dt, max(t_stage_end - t, p_nd.dt_min))
        q = mult * Q2D
        set_volumetric_source!(stokes.Q, Q_buffer, ind, q * dt / source_area)

        Pf_used, λ_used = prescribed_pore_pressure(
            ustrip.(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), C, CD)),
            Array(Plitho), Array(phase_ratios.center), Array(ϕ_R.center), p_nd,
        )
        args = (; T = thermal.T, Pf = PTArray(backend)(p.α_B .* Pf_used), P = stokes.P, dt = Inf, ΔT = thermal.ΔT)
        stress2grid!(stokes, pτ, particles)
        solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ_R, rheology, args, grid, dt, igg;
            kwargs = (;
                air_phase = AIR_PHASE,
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 100.0e3,
                total_iterMax = 100.0e3,
                nout = 50,
                rel_drop = 1.0e-2,
                λ_relaxation_PH = 1.0,
                λ_relaxation_DR = 1.0,
                pressure_relaxation = 1,
                viscosity_relaxation = 1.0e-3,
                viscosity_cutoff = cutoff_visc,
                free_surface = true,
            ),
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)
        A_injected += q * dt

        # Diagnostics at the end of this step (current geometry)
        lithostatic_pressure!(Plitho, ρg_litho, ρg[2], ϕ_R, grid, igg)
        diag = failure_diagnostics(
            stokes, thermal.T, Plitho, P_start, T_start, phase_ratios, ϕ_R, xci_cpu, cap, p_nd, CD;
            chamber = chamber, Pf_used, λ_used,
        )
        uplift = maximum(ustrip.(dimensionalize(Array(chain.h_vertices) .- h0, m, CD)))

        if onset === nothing && diag.halo_yield
            onset = (
                ; x = diag.x_yield, y = diag.y_yield, t = t + dt, it = it + 1,
                mode = diag.yield_mode, λ = diag.lambda_yield,
            )
            @printf(
                "\n>>> Halo yielding at t = %.4f yr (it %d): x = %.2f km, y = %.2f km, mode = %s, λ = %.2f, ΔP = %.2f MPa\n\n",
                ustrip(dimensionalize(t + dt, yr, CD)), it + 1,
                ustrip(dimensionalize(diag.x_yield, km, CD)), ustrip(dimensionalize(diag.y_yield, km, CD)),
                diag.yield_mode, diag.lambda_yield, ustrip(dimensionalize(diag.ΔP_chamber, MPa, CD)),
            )
        end

        compute_shear_heating!(thermal, stokes, phase_ratios, rheology, dt)
        dt_next = min(compute_dt(stokes, di_min, dt_max) * 0.95, dt_max)
        dt_next = failure_limited_dt(dt_next, dt, diag.r_max_eff, r_old, p_nd)
        r_old = diag.r_max_eff

        rotate_stress!(pτ, stokes, particles, dt)

        heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
            kwargs = (; igg = igg, phase = phase_ratios, iterMax = 10.0e3, nout = 1.0e2, verbose = false),
        )
        shear_output = if do_vtk && igg.me == 0 && (it == 0 || rem(it + 1, p.nout) == 0 || (onset !== nothing && onset.it == it + 1))
            shear_heating_output(thermal, stokes.P, phase_ratios.center, rheology, dt, CD)
        else
            nothing
        end
        subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes)
        @views dt₀[1, :] .= dt₀[2, :]
        @views dt₀[end, :] .= dt₀[end - 1, :]
        @views dt₀[:, 1] .= dt₀[:, 2]
        @views dt₀[:, end] .= dt₀[:, end - 1]
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt)

        # Advection
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        semilagrangian_advection_markerchain!(
            chain, RungeKutta4(), @velocity(stokes), grid.xi_vel, xvi, dt; conserve_mean = false
        )
        update_phases_given_markerchain!(pPhases, chain, particles, origin, grid.di.vertex, AIR_PHASE)
        inject_particles_phase!(
            particles, pPhases, particle_args_reduced,
            (thermal.T, stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c, stokes.ω.xy_c),
        )
        update_phases_given_markerchain!(pPhases, chain, particles, origin, grid.di.vertex, AIR_PHASE)
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ_R, chain, xvi, grid.di.vertex)
        @parallel (@idx size(particles.index)) set_air_particle_temperature!(
            pT, pPhases, particles.index, AIR_PHASE, Ttop,
        )

        it += 1
        t += dt
        particle2centroid!(thermal.T, pT, particles)

        # Output
        write_timeseries_row(ts_io, it, t, dt, mult, A_injected, diag, uplift, CD)
        push!(
            rows, (;
                t = ustrip(dimensionalize(t, yr, CD)),
                ΔP = ustrip(dimensionalize(diag.ΔP_chamber, MPa, CD)),
                uplift, r = diag.r_max, r_eff = diag.r_max_eff, r_undrained = diag.r_max_undrained,
                halo_yield = diag.halo_yield,
            )
        )
        @printf(
            "it %5d | t = %10.3f yr | dt = %9.4f yr | Q× %4.1f | ΔP = %7.3f MPa | uplift = %7.3f m | r = %.3f (eff %.3f, undrained %.3f)\n",
            it, ustrip(dimensionalize(t, yr, CD)), ustrip(dimensionalize(dt, yr, CD)), mult,
            ustrip(dimensionalize(diag.ΔP_chamber, MPa, CD)), uplift, diag.r_max, diag.r_max_eff,
            diag.r_max_undrained,
        )

        if igg.me == 0 && (it == 1 || rem(it, p.nout) == 0 || (onset !== nothing && onset.it == it))
            plot_snapshot(figdir, it, t, stokes, thermal, Plitho, chain, xci_cpu, xvi_cpu, li, CD, onset)
            plot_timeseries(figdir, rows)
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, MPa, CD))),
                    Vx = Array(ustrip.(dimensionalize(Vx_v, cm / yr, CD))),
                    Vy = Array(ustrip.(dimensionalize(Vy_v, cm / yr, CD))),
                )
                data_c = (;
                    phase = [argmax(pr) for pr in Array(phase_ratios.center)],
                    P = Array(ustrip.(dimensionalize(stokes.P, MPa, CD))),
                    overP = Array(ustrip.(dimensionalize(stokes.P .- Plitho, MPa, CD))),
                    shear_output...,
                    softened_strength_output(stokes.EII_pl, phase_ratios.center, rheology, CD)...,
                    Pf = diag.Pf,
                    λ_pf = diag.λ_field,
                    dp_undrained = diag.Δp_undrained,
                    T = Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), 2:(end - 1)], C, CD))),
                    τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CD))),
                    εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1, CD))),
                    εII_pl = Array(ustrip.(dimensionalize(stokes.ε_pl.II, s^-1, CD))),
                    EII_pl = Array(stokes.EII_pl),
                    EVol_pl = Array(stokes.EVol_pl),
                    εvol_pl = Array(ustrip.(dimensionalize(stokes.ε_vol_pl, s^-1, CD))),
                    plastic_multiplier = Array(ustrip.(dimensionalize(stokes.λ, s^-1, CD))),
                    η_vep = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CD))),
                    ϕ_R = Array(ϕ_R.center),
                )
                velocity_v = (
                    Array(ustrip.(dimensionalize(Vx_v, cm / yr, CD))),
                    Array(ustrip.(dimensionalize(Vy_v, cm / yr, CD))),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    Array.(grid.xvi), Array.(grid.xci), data_v, data_c, velocity_v;
                    t = ustrip(dimensionalize(t, yr, CD)) / 1.0e3,
                )
                save_marker_chain(
                    joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")),
                    Array(grid.xvi[1]), Array(chain.h_vertices),
                )
            end
        end

        dt = dt_next
    end

    close(ts_io)
    plot_timeseries(figdir, rows)
    return onset
end

# -----------------------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------------------
n = 128
nx = n
ny = n
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# Reference case: no fault, no background strain, magma K = 5 GPa.
onset = main(igg, nx, ny; 
    do_vtk = true,
    λ_mode = :thermal,
    # λ_mode = :uniform,
    figdir = "RF2D_thermal",
    λ_pf = 1.0,
    )

# Examples for the parameter study:
# main(igg, nx, ny; figdir = "RF2D_fault_ext", has_fault = true, εbg = 1.0e-15 / s)
# main(igg, nx, ny; figdir = "RF2D_K1GPa", K_magma = 1.0GPa)
# main(igg, nx, ny; figdir = "RF2D_fastQ", Q2D = 1.0e-3km^2 / yr)
# main(igg, nx, ny; figdir = "RF2D_linear", linear = true)

# onset = main(
#     igg, nx, ny;
#     do_vtk = true,
#     figdir = "RF2D_tensile_probe_thermal",
#     linear = true,
#     λ_mode = :thermal,
#     # λ_mode = :uniform,
#     λ_pf = 1.0,
#     pT = -1.0MPa,
#     C = 20.0MPa,
#     Ψ = 0.0,
#     has_fault = false,
#     εbg = 0.0 / s,
#     Q2D = 1.0e-3km^2 / yr,
#     schedule = ((100.0yr, 1.0),),
#     t_end = 100.0yr,
#     dt_ini = 0.1yr,
#     dt_max = 1.0yr,
# )
