# Shared setup for the brittle-ductile transition benchmark of
# Popov, Berlie and Kaus (2025), Sect. 4.5 and Fig. 9.
#
# Included by `BrittleDuctile2D.jl` (pseudo-transient Stokes, sticky air) and
# `BrittleDuctile2D_VariationalDYREL.jl` (variational DYREL, marker-chain free surface).
# Include it after `@init_parallel_stencil`, because it defines kernels.
#
# Phase 1 is air, phase 2 is the crust, which is the numbering the variational solvers
# expect through their `air_phase` keyword.

# PARAMETERS (Popov et al. 2025, Table 1, "Brittle-ductile" column) ---------------
const ρ_rock = 3.0e3        # kg/m^3
const B_D = 5.0e-24         # Pa^-1 s^-1, linear creep; Table 1 lists no activation energy
const B_N = 8.8971e-25      # Pa^-n s^-1, dislocation creep prefactor
const E_N = 1.9e5           # J/mol
const n_creep = 3.3
const G_rock = 5.0e10       # Pa
const K_rock = 1.1e11       # Pa
const ϕ_fric = 30.0         # degrees
const Ψ_dil = 3.0           # degrees
const C_init = 2.0e7        # Pa
const C_min = 5.0e6         # Pa
const H_c = -5.0e8          # Pa, cohesion softening modulus
const p_T = -1.0e6          # Pa, tensile strength
const η_vp = 1.0e19         # Pa s, Perzyna regularization
const εbg = 1.0e-15         # 1/s, horizontal extension rate
const geotherm = 20.0e-3    # K/m
const T_surf = 273.15       # K
const R_gas = 8.31446
const g_grav = 9.81

const yr = 3600 * 24 * 365.25
const air_phase = 1

"""
    rheology_setup(; with_cap, η_air)

Air (phase 1) and crust (phase 2). `with_cap = false` swaps the tensile cap for a plain
regularized Drucker-Prager cone and is the control the benchmark compares against.
"""
function rheology_setup(; with_cap::Bool, η_air = 1.0e20)
    el = ConstantElasticity(; G = G_rock, Kb = K_rock)
    # cohesion falls from C_init to C_min at the rate H_c
    soft_C = LinearSoftening((C_min, C_init), (0.0, (C_init - C_min) / abs(H_c)))
    pl = if with_cap
        DruckerPragerCap(; C = C_init, ϕ = ϕ_fric, η_vp = η_vp, Ψ = Ψ_dil, pT = p_T, softening_C = soft_C)
    else
        DruckerPrager_regularised(; C = C_init, ϕ = ϕ_fric, η_vp = η_vp, Ψ = Ψ_dil, softening_C = soft_C)
    end
    # `Apparatus = Invariant` keeps GeoParams from applying its uniaxial-to-invariant
    # correction (a factor 5.3 at n = 3.3); Table 1 is already in invariant SI form.
    disl = DislocationCreep(;
        A = B_N / s / Pa^n_creep, n = n_creep, E = E_N * J / mol,
        V = 0.0 * m^3 / mol, r = 0.0, R = R_gas * J / mol / K, Apparatus = GeoParams.Invariant,
    )
    lin = LinearViscous(; η = inv(2 * B_D) * Pa * s)

    air = SetMaterialParams(;
        Phase = 1,
        Density = ConstantDensity(; ρ = 1.0),
        Gravity = ConstantGravity(; g = g_grav),
        CompositeRheology = CompositeRheology((LinearViscous(; η = η_air),)),
    )
    crust = SetMaterialParams(;
        Phase = 2,
        Density = ConstantDensity(; ρ = ρ_rock),
        Gravity = ConstantGravity(; g = g_grav),
        CompositeRheology = CompositeRheology((lin, disl, el, pl)),
        Elasticity = el,
    )
    return (air, crust)
end

# geotherm below the surface, isothermal air above it; `T` carries a ghost ring
@parallel_indices (i, j) function init_T!(T, y, air_top)
    depth = air_top - y[j]
    @inbounds T[i + 1, j + 1] = depth > 0 ? T_surf + geotherm * depth : T_surf
    return nothing
end

"""
    strength_envelope(depth)

Far-field strength at `depth`: the smaller of the Drucker-Prager envelope and the stress
that drives the imposed background strain rate through the linear and dislocation creep
laws. The solved far-field column is compared against this, and its peak must sit inside
the domain for the model to say anything about the brittle-ductile transition.
"""
function strength_envelope(depth)
    T = T_surf + geotherm * depth
    P = ρ_rock * g_grav * depth
    brittle = C_init * cosd(ϕ_fric) + P * sind(ϕ_fric)
    f(τ) = B_D * τ + B_N * exp(-E_N / (R_gas * T)) * τ^n_creep - εbg
    lo, hi = 1.0, 1.0e12
    for _ in 1:200
        mid = sqrt(lo * hi)
        f(mid) > 0 ? (hi = mid) : (lo = mid)
    end
    return min(brittle, sqrt(lo * hi))
end

"""
    plastic_strain_seed(xci, ni, lx, air_top; seed, amplitude, halfwidth, depth)

Random accumulated plastic strain in the central upper crust. Popov et al. seed
localization this way so that cohesion softening has somewhere to start.
"""
function plastic_strain_seed(
        xci, ni, lx, air_top; seed = 1234, amplitude = 0.02,
        halfwidth = 20.0e3, depth = 7.0e3,
    )
    field = zeros(ni...)
    rng = MersenneTwister(seed)
    for j in axes(field, 2), i in axes(field, 1)
        x, y = xci[1][i], xci[2][j]
        if abs(x - 0.5lx) < halfwidth && (air_top - depth) < y < air_top
            field[i, j] = amplitude * rand(rng)
        end
    end
    return field
end

"""
    report(results)

Print one row per case. The cap run must open near-surface tensile zones that the
control cannot; everything else in the table is context for that comparison.
"""
function report(results)
    println(
        "\n case    τ peak [MPa] @ depth [km]   analytic peak   EII_max    EVol near surface   unconverged steps"
    )
    for r in results
        println(
            rpad(r.with_cap ? "cap" : "nocap", 8),
            rpad(round(r.τ_peak_MPa; digits = 1), 9), " @ ", rpad(round(r.τ_peak_depth_km; digits = 1), 10),
            rpad(round(r.envelope_peak_MPa; digits = 1), 16),
            rpad(round(r.EII_max; sigdigits = 3), 11),
            rpad(round(r.EVol_near_surface; sigdigits = 3), 20),
            isempty(r.unconverged) ? "none" : string(r.unconverged),
        )
    end
    return println(
        "\nRead the EVol columns and the figures first: a run with unconverged steps says nothing."
    )
end

"""
    diagnostics(stokes, xci, nx, air_top; with_cap, nx_probe)

Far-field stress column against the analytical envelope, plus the plastic-strain
diagnostics the benchmark is judged on.
"""
function diagnostics(stokes, xci, nx, air_top, nsteps, unconverged; with_cap, ny)
    depths = [air_top - y for y in xci[2] if y < air_top]
    idx = max(1, nx ÷ 20)     # far from the seeded central zone
    τ_column = [Array(stokes.τ.II)[idx, j] for (j, y) in enumerate(xci[2]) if y < air_top]
    envelope = strength_envelope.(depths)
    EVol = Array(stokes.EVol_pl)
    EII = Array(stokes.EII_pl)
    near_surface = [j for (j, y) in enumerate(xci[2]) if (air_top - 3.0e3) < y < air_top]
    return (;
        with_cap, nx, ny, nsteps, unconverged, depths, τ_column, envelope, EVol, EII,
        τ_peak_MPa = maximum(τ_column) / 1.0e6,
        τ_peak_depth_km = depths[argmax(τ_column)] / 1.0e3,
        envelope_peak_MPa = maximum(envelope) / 1.0e6,
        envelope_peak_depth_km = depths[argmax(envelope)] / 1.0e3,
        EII_max = maximum(EII),
        EVol_max = maximum(EVol),
        EVol_near_surface = isempty(near_surface) ? 0.0 : maximum(EVol[:, near_surface]),
    )
end

"""
    save_figure(figdir, r, xci, tag)

Shear bands, tensile zones and the strength envelope of one case.
"""
function save_figure(figdir, r, xci, tag)
    take(figdir)
    fig = Figure(size = (1600, 900))
    ax1 = Axis(fig[1, 1], title = L"E_{II}^{pl}", xlabel = "x [km]", ylabel = "y [km]")
    ax2 = Axis(fig[2, 1], title = L"E_{Vol}^{pl}", xlabel = "x [km]", ylabel = "y [km]")
    ax3 = Axis(fig[1:2, 2], title = "strength envelope", xlabel = "τII [MPa]", ylabel = "depth [km]")
    heatmap!(ax1, xci[1] ./ 1.0e3, xci[2] ./ 1.0e3, r.EII, colormap = :batlow)
    heatmap!(ax2, xci[1] ./ 1.0e3, xci[2] ./ 1.0e3, r.EVol, colormap = :batlow)
    lines!(ax3, r.envelope ./ 1.0e6, r.depths ./ 1.0e3, color = :red, label = "analytical")
    scatter!(ax3, r.τ_column ./ 1.0e6, r.depths ./ 1.0e3, color = :black, markersize = 5, label = "model")
    ax3.yreversed = true
    axislegend(ax3)
    return save(joinpath(figdir, "brittle_ductile_$(tag)_$(r.nx)x$(r.ny).png"), fig)
end
