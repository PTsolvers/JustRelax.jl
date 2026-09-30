# =========================================================================================
# E0_DeborahIsotherm.jl
#
# Deborah-isotherm prediction (reservoir_failure_paper_plan.md, §1.2 and §6.0).
#
# Predicts where injection-induced yielding should start around the chamber of
# `ReservoirFailure2D.jl`, before any production run: on the surface where the local
# Maxwell time equals the loading time,
#
#     τ_M(x) = η_eff(T(x), τ_II(x)) / G_rock  ≈  τ_load.
#
#   * T(x): pre-heating conduction with the same geometry, geotherm, material
#     properties and fixed magma temperature as `ReservoirFailure2D.jl` (backward Euler,
#     finite volumes on a uniform grid; the chamber is a fixed-temperature region).
#   * τ_II(x) = ΔP_ref (a / r)^2: elastic plane-strain (Lamé) deviatoric stress around a
#     pressurized cylinder of radius a. It ignores the free surface and relaxation, so it
#     is a stress scale, not a solution; ΔP_ref is bracketed.
#   * η_eff: the host dislocation-creep law of `ReservoirFailure2D.jl` (GeoParams
#     convention ε̇_II = A (F_T τ_II)^n exp(-E/RT) / F_E), clipped to the solver's
#     viscosity cutoffs.
#   * τ_load = (σ_ref / K_eff) A_ch / Q2D with 1/K_eff = 1/K_magma + 1/G_rock
#     (plan §6.2), σ_ref = cohesion C.
#
# Along rays from the chamber centre (θ measured from the apex), the predicted onset is
# the first point, moving outward from the wall, where τ_M/τ_load ≥ 1. "wall" means
# the ratio already exceeds 1 at the wall (elastic limit); "none" means it stays below 1
# within `r_search` of the wall. The halo thickness δ_T is the distance from the wall to
# the `T_halo` isotherm along the same ray.
#
# Material and geometry values mirror the defaults of `ReservoirFailure2D.jl`; update
# both together. Temperature- and pressure-dependent density is replaced by the
# reference densities in the heat capacity term.
#
# Output (in `outdir`): predictions.csv, metadata.txt (date, git commit, parameters) and
# figures. Archive the output before E1 runs.
# =========================================================================================

using Pkg
Pkg.activate(@__DIR__)
using GeoParams, SparseArrays, LinearAlgebra, Dates, Printf

Pkg.activate(joinpath(@__DIR__, "miniapps"))
using CairoMakie

const yr = 365.25 * 24 * 3600.0
const kyr = 1.0e3 * yr
const km = 1.0e3

default_E0_params() = (;
    # geometry (ReservoirFailure2D.jl defaults)
    Lx = 50.0km,
    D = 30.0km,
    chamber_depth = 5.0km,
    chamber_radius = 1.5km,
    dx = 100.0,                    # uniform cell size [m]
    # thermal
    T_top = 20.0,                  # °C
    T_bot = 438.5625,              # °C at depth D
    k_rock = 3.0,                  # W/m/K
    ρ_rock = 2650.0,
    Cp = 1050.0,
    H_r = 1.0e-6,                  # W/m^3
    dt_thermal = 250.0yr,
    # rock mechanics
    G_rock = 30.0e9,
    K_magma = 5.0e9,
    C = 20.0e6,                    # σ_ref for τ_load
    η_cutoff = (1.0e16, 1.0e24),   # Pa s, as in ReservoirFailure2D.jl
    # sweep
    t_preheat = (5.0, 20.0, 50.0, 200.0) .* kyr,
    T_magma = (900.0, 1100.0, 1300.0),
    Q2D = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2) .* (km^2 / yr),
    dP_ref = (5.0e6, 10.0e6, 20.0e6),
    θ = 0.0:15.0:180.0,            # degrees from the apex
    # diagnostics
    T_halo = 400.0,                # °C, isotherm defining δ_T
    r_search = 10.0km,             # search distance beyond the wall
    dr = 10.0,                     # ray sampling step [m]
    # reference case for the maps
    ref = (; t_preheat = 50.0kyr, T_magma = 1300.0, Q2D = 1.0e-4km^2 / yr, dP_ref = 10.0e6),
    outdir = joinpath(@__DIR__, "E0_DeborahIsotherm"),
)

# -----------------------------------------------------------------------------------------
# Host creep law: identical to `creep_models()` in ReservoirFailure2D.jl
# -----------------------------------------------------------------------------------------
host_creep() = DislocationCreep(;
    n = 3.3NoUnits,
    A = 1.0 * exp10(-5.7)MPa^(-33 // 10) / s,
    E = 186.5kJ / mol,
    V = 0m^3 / mol,
    r = 0NoUnits,
    Apparatus = AxialCompression,
)

"""
    creep_coefficients(cr)

Plain-number coefficients of a GeoParams `DislocationCreep` with A in MPa^-n s^-1.
"""
function creep_coefficients(cr)
    n = Float64(NumValue(cr.n))
    GeoParams.Unit(cr.A) == MPa^(-33 // 10) / s || error("unexpected unit of A; expected MPa^-3.3 s^-1")
    E = if GeoParams.Unit(cr.E) == kJ / mol
        1.0e3 * NumValue(cr.E)
    elseif GeoParams.Unit(cr.E) == J / mol
        Float64(NumValue(cr.E))
    else
        error("unexpected unit of E; expected kJ/mol or J/mol")
    end
    GeoParams.Unit(cr.R) == J / mol / K || error("unexpected unit of R; expected J/mol/K")
    return (; n, A = Float64(NumValue(cr.A)), E, R = Float64(NumValue(cr.R)), FT = Float64(cr.FT), FE = Float64(cr.FE))
end

"Effective viscosity [Pa s] at deviatoric stress τ [Pa] and temperature T [°C]."
function η_eff(c, τ, T, cutoff)
    τ > 0 || throw(ArgumentError("stress must be positive, got $τ"))
    τMPa = τ / 1.0e6
    εII = c.A * (c.FT * τMPa)^c.n * exp(-c.E / (c.R * (T + 273.15))) / c.FE
    return clamp(τ / (2εII), cutoff...)
end

# -----------------------------------------------------------------------------------------
# Pre-heating conduction
# -----------------------------------------------------------------------------------------
function thermal_grid(p)
    nx = round(Int, p.Lx / p.dx)
    ny = round(Int, p.D / p.dx)
    nx * p.dx ≈ p.Lx && ny * p.dx ≈ p.D || throw(ArgumentError("dx must divide Lx and D"))
    xc = [-p.Lx / 2 + (i - 0.5) * p.dx for i in 1:nx]
    yc = [-p.D + (j - 0.5) * p.dx for j in 1:ny]
    return xc, yc
end

in_chamber(x, y, p) = x^2 + (y + p.chamber_depth)^2 ≤ p.chamber_radius^2

"""
    conduction_operator(p, xc, yc)

Backward-Euler matrix for ρCp ∂T/∂t = ∇·(k∇T) + H_r with Dirichlet top/bottom
(half-cell distance), no-flux sides, and chamber cells held at a fixed temperature.
"""
function conduction_operator(p, xc, yc)
    nx, ny = length(xc), length(yc)
    L = LinearIndices((nx, ny))
    c = p.ρ_rock * p.Cp * p.dx^2 / (p.k_rock * p.dt_thermal)
    I, J, V = Int[], Int[], Float64[]
    for j in 1:ny, i in 1:nx
        row = L[i, j]
        if in_chamber(xc[i], yc[j], p)
            push!(I, row); push!(J, row); push!(V, 1.0)
            continue
        end
        diag = c
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ii, jj = i + di, j + dj
            if 1 ≤ ii ≤ nx && 1 ≤ jj ≤ ny
                push!(I, row); push!(J, L[ii, jj]); push!(V, -1.0)
                diag += 1.0
            elseif jj < 1 || jj > ny
                diag += 2.0            # Dirichlet boundary half a cell away
            end                        # sides: no flux
        end
        push!(I, row); push!(J, row); push!(V, diag)
    end
    return lu(sparse(I, J, V, nx * ny, nx * ny)), c
end

function rhs!(b, T, Tmagma, p, xc, yc, c)
    nx, ny = length(xc), length(yc)
    L = LinearIndices((nx, ny))
    src = p.H_r * p.dx^2 / p.k_rock
    for j in 1:ny, i in 1:nx
        row = L[i, j]
        if in_chamber(xc[i], yc[j], p)
            b[row] = Tmagma
            continue
        end
        b[row] = c * T[row] + src
        j == 1 && (b[row] += 2.0 * p.T_bot)
        j == ny && (b[row] += 2.0 * p.T_top)
    end
    return b
end

"Temperature fields [°C] at each requested pre-heating time, for one magma temperature."
function preheat_fields(p, F, c, xc, yc, Tmagma)
    nx, ny = length(xc), length(yc)
    T = [in_chamber(x, y, p) ? Tmagma : p.T_top + (p.T_bot - p.T_top) * (-y / p.D) for x in xc, y in yc]
    T = vec(T)
    b = similar(T)
    targets = sort(collect(p.t_preheat))
    steps = round.(Int, targets ./ p.dt_thermal)
    all(steps .* p.dt_thermal .≈ targets) || throw(ArgumentError("dt_thermal must divide every t_preheat"))
    out = Dict{Float64, Matrix{Float64}}()
    for n in 1:maximum(steps)
        T = F \ rhs!(b, T, Tmagma, p, xc, yc, c)
        k = findfirst(==(n), steps)
        k === nothing || (out[targets[k]] = reshape(copy(T), nx, ny))
    end
    return out
end

# -----------------------------------------------------------------------------------------
# Deborah ratio and rays
# -----------------------------------------------------------------------------------------
τ_load(p, Q2D) = (p.C / inv(inv(p.K_magma) + inv(p.G_rock))) * (π * p.chamber_radius^2) / Q2D

"Bilinear interpolation of a cell-centred field on the uniform grid."
function interp(T, xc, yc, x, y)
    dx = xc[2] - xc[1]
    fi = (x - xc[1]) / dx + 1
    fj = (y - yc[1]) / dx + 1
    i = clamp(floor(Int, fi), 1, length(xc) - 1)
    j = clamp(floor(Int, fj), 1, length(yc) - 1)
    s, t = clamp(fi - i, 0, 1), clamp(fj - j, 0, 1)
    return (1 - s) * (1 - t) * T[i, j] + s * (1 - t) * T[i + 1, j] + (1 - s) * t * T[i, j + 1] + s * t * T[i + 1, j + 1]
end

"""
    ray_prediction(T, xc, yc, p, cc, θ, dP, tload)

Distance from the wall [m] of the first point with τ_M/τ_load ≥ 1 along the ray at θ
(degrees from the apex), the temperature there [°C], the halo thickness δ_T and the
onset class.
"""
function ray_prediction(T, xc, yc, p, cc, θ, dP, tload)
    a = p.chamber_radius
    yc0 = -p.chamber_depth
    ex, ey = sind(θ), cosd(θ)
    d_onset = NaN
    T_onset = NaN
    δT = NaN
    for d in 0.0:p.dr:p.r_search
        r = a + d
        x, y = r * ex, yc0 + r * ey
        y ≥ 0 && break                       # reached the surface
        Tloc = interp(T, xc, yc, x, y)
        isnan(δT) && Tloc < p.T_halo && (δT = d)
        if isnan(d_onset)
            τ = dP * (a / r)^2
            ratio = η_eff(cc, τ, Tloc, p.η_cutoff) / p.G_rock / tload
            if ratio ≥ 1
                d_onset = d
                T_onset = Tloc
            end
        end
        !isnan(d_onset) && !isnan(δT) && break
    end
    class = isnan(d_onset) ? "none" : (d_onset == 0 ? "wall" : "off-wall")
    return d_onset, T_onset, δT, class
end

function deborah_map(T, xc, yc, p, cc, dP, tload)
    a = p.chamber_radius
    return [
        let r = hypot(x, y + p.chamber_depth)
            r ≤ a ? NaN : log10(η_eff(cc, dP * (a / r)^2, T[i, j], p.η_cutoff) / p.G_rock / tload)
        end
            for (i, x) in pairs(xc), (j, y) in pairs(yc)
    ]
end

# -----------------------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------------------
function git_commit()
    try
        return readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        return "unknown (git unavailable)"
    end
end

function run_E0(; kw...)
    p = merge(default_E0_params(), (; kw...))
    mkpath(p.outdir)
    cc = creep_coefficients(host_creep())
    xc, yc = thermal_grid(p)
    F, c = conduction_operator(p, xc, yc)

    fields = Dict(Tm => preheat_fields(p, F, c, xc, yc, Tm) for Tm in p.T_magma)

    open(joinpath(p.outdir, "predictions.csv"), "w") do io
        println(io, "t_preheat_kyr,T_magma_C,Q2D_km2_per_yr,dP_ref_MPa,tau_load_yr,theta_deg,d_f_star_km,T_onset_C,delta_T_km,d_over_delta,class")
        for Tm in p.T_magma, tp in p.t_preheat, Q in p.Q2D, dP in p.dP_ref, θ in p.θ
            tl = τ_load(p, Q)
            d, Tf, δ, cls = ray_prediction(fields[Tm][tp], xc, yc, p, cc, θ, dP, tl)
            @printf(
                io, "%g,%g,%g,%g,%.6g,%g,%.4g,%.4g,%.4g,%.4g,%s\n",
                tp / kyr, Tm, Q / (km^2 / yr), dP / 1.0e6, tl / yr, θ, d / km, Tf, δ / km, d / δ, cls
            )
        end
    end

    open(joinpath(p.outdir, "metadata.txt"), "w") do io
        println(io, "E0 Deborah-isotherm prediction")
        println(io, "created: ", now())
        println(io, "git commit: ", git_commit())
        println(io, "creep coefficients (A in MPa^-n s^-1): ", cc)
        for (k, v) in pairs(p)
            k === :outdir && continue
            println(io, k, " = ", v)
        end
    end

    plot_reference(p, fields, xc, yc, cc)
    plot_summary(p, fields, xc, yc, cc)
    return p.outdir
end

function plot_reference(p, fields, xc, yc, cc)
    r = p.ref
    T = fields[r.T_magma][r.t_preheat]
    tl = τ_load(p, r.Q2D)
    D = deborah_map(T, xc, yc, p, cc, r.dP_ref, tl)
    win_x = abs.(xc) .≤ 8km
    win_y = -12km .≤ yc .≤ 0
    fig = Figure(size = (900, 520))
    ax = Axis(
        fig[1, 1]; aspect = DataAspect(), xlabel = "x [km]", ylabel = "y [km]",
        title = @sprintf(
            "log10(τ_M/τ_load): t_preheat = %g kyr, T_magma = %g °C, Q2D = %g km²/yr, ΔP_ref = %g MPa",
            r.t_preheat / kyr, r.T_magma, r.Q2D / (km^2 / yr), r.dP_ref / 1.0e6
        )
    )
    hm = heatmap!(ax, xc[win_x] ./ km, yc[win_y] ./ km, D[win_x, win_y]; colormap = :vik, colorrange = (-4, 4))
    contour!(ax, xc[win_x] ./ km, yc[win_y] ./ km, D[win_x, win_y]; levels = [0.0], color = :black, linewidth = 2)
    contour!(ax, xc[win_x] ./ km, yc[win_y] ./ km, T[win_x, win_y]; levels = 200:100:1200, color = :gray40, linestyle = :dash)
    Colorbar(fig[1, 2], hm; label = "log10(τ_M/τ_load)  (black: = 0; dashed: isotherms every 100 °C)")
    save(joinpath(p.outdir, "deborah_map_reference.png"), fig)
    return fig
end

function plot_summary(p, fields, xc, yc, cc)
    fig = Figure(size = (1100, 380))
    for (k, Tm) in enumerate(p.T_magma)
        ax = Axis(
            fig[1, k]; xscale = log10, xlabel = "Q2D [km²/yr]", ylabel = "d_f* at θ = 90° [km]",
            title = @sprintf("T_magma = %g °C, ΔP_ref = %g MPa", Tm, p.ref.dP_ref / 1.0e6)
        )
        for tp in p.t_preheat
            d = [ray_prediction(fields[Tm][tp], xc, yc, p, cc, 90.0, p.ref.dP_ref, τ_load(p, Q))[1] / km for Q in p.Q2D]
            scatterlines!(ax, collect(p.Q2D) ./ (km^2 / yr), d; label = @sprintf("%g kyr", tp / kyr))
        end
        k == 1 && axislegend(ax; position = :lt)
    end
    save(joinpath(p.outdir, "onset_distance_summary.png"), fig)
    return fig
end

# run_E0()
