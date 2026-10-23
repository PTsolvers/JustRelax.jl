# Response of Santorini's shallow magma chamber to sea-level change, after
# Satow et al. (2021), Nature Geoscience 14, 586–592, doi:10.1038/s41561-021-00783-4
# (Methods, "Modelling the effect of Quaternary sea-level changes").
#
# - 100 km wide, 20 km thick crust; free-slip lateral edges and base, free surface.
# - Five 500 m thick layers (E = 5, 10, 15, 20, 30 GPa) above a host unit
#   (E = 40 GPa); ν = 0.25, ρ = 2700 kg/m³. The crust is visco-elastic, with
#   plagioclase (An75) dislocation creep (Ji & Zhao, 1993)
#   and Drucker–Prager plasticity (C = 10 MPa, ϕ = 30°). Plastic yielding sees
#   the effective pressure P - Pf, with pore pressure Pf dry (default),
#   hydrostatic or lithostatic (`pore_pressure`).
# - Sill-like elliptical magma chamber, 6 km wide and 1 km thick, centred at
#   4 km depth (dimensions read from Fig. 2), filled with a low-viscosity,
#   compressible mush held at 1100 °C by a Dirichlet thermal condition.
# - Linear background geotherm, 20 °C at the surface and 30 °C/km; heat diffuses
#   from the chamber into the crust as the model evolves.
# - The only load is the change of seawater pressure on the surface,
#   Δp = ρw g (h - h_ref) (ρw = 1025 kg/m³, g = 9.8 m/s²), relative to the sea
#   level h_ref at which the model starts in equilibrium. Gravity is on and the
#   initial pressure is lithostatic; the tensile stresses plotted are the change
#   from that initial state.
#
# Two sea-level histories (`history` below):
# - `:steps`: 10 m drops from 0 m to −110 m, 1 kyr apart, as in Fig. 2.
# - `:Grant2014`: the Red Sea relative sea-level record (probability maximum) of
#   Grant et al. (2014), read from `Grant2014_RedSea_RSL.csv`, which Satow et
#   al. (2021) align with Santorini's eruptions (Fig. 4).
#
# The surface load is imposed through `surface_pressure` on the embedded
# rock–air interface of the variational Stokes solver. The first principal
# stress at the chamber wall is printed and plotted against time; Supplementary
# Fig. 2B of Satow et al. (2021) reports ≈3.5 MPa at −40 m for their cavity model.

# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg;
Pkg.activate("miniapps")

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

using GeoParams, Printf, GLMakie, PoissonGrids

# Phase indices
const n_layers = 5           # stratified layers above the host unit
const host_phase = n_layers + 1
const magma_phase = n_layers + 2
const air_phase = n_layers + 3

function init_phases!(phases, particles, layer_thickness, xc_chamber, yc_chamber, a_chamber, b_chamber)
    ni = size(phases)

    @parallel_indices (i, j) function _init_phases!(
            phases, px, py, index, layer_thickness, xc_chamber, yc_chamber, a_chamber, b_chamber
        )
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = @index py[ip, i, j]
            depth = -y

            phase = if depth < 0
                air_phase
            elseif ((x - xc_chamber) / a_chamber)^2 + ((y - yc_chamber) / b_chamber)^2 ≤ 1
                magma_phase
            else
                min(floor(Int, depth / layer_thickness) + 1, host_phase)
            end
            @index phases[ip, i, j] = Float64(phase)
        end
        return nothing
    end

    return @parallel (@idx ni) _init_phases!(
        phases, particles.coords..., particles.index,
        layer_thickness, xc_chamber, yc_chamber, a_chamber, b_chamber,
    )
end

# Rock is a Maxwell material with power-law creep in series with elasticity and
# Drucker–Prager plasticity.
function init_rheology(
        CD; E_layers, E_host, ν, ρ_rock, creep_rock, plasticity, k_rock, Cp,
        ρ_magma, η_magma, G_magma, k_magma, η_air,
    )
    g = 9.81m / s^2
    rock(phase, E) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ = ρ_rock),
        HeatCapacity = ConstantHeatCapacity(; Cp),
        Conductivity = ConstantConductivity(; k = k_rock),
        CompositeRheology = CompositeRheology((creep_rock, SetConstantElasticity(; E, ν), plasticity)),
        Elasticity = SetConstantElasticity(; E, ν),
        Gravity = ConstantGravity(; g),
        CharDim = CD,
    )
    layers = ntuple(k -> rock(k, E_layers[k]), n_layers)
    host = rock(host_phase, E_host)

    # Low-viscosity mush: viscous in shear, elastic in bulk.
    el_magma = SetConstantElasticity(; G = G_magma, ν)
    magma = SetMaterialParams(;
        Phase = magma_phase,
        Density = ConstantDensity(; ρ = ρ_magma),
        HeatCapacity = ConstantHeatCapacity(; Cp),
        Conductivity = ConstantConductivity(; k = k_magma),
        CompositeRheology = CompositeRheology((LinearViscous(; η = η_magma), el_magma)),
        Elasticity = el_magma,
        Gravity = ConstantGravity(; g),
        CharDim = CD,
    )

    # Air: large heat capacity and conductivity keep it at the surface temperature.
    el_air = SetConstantElasticity(; E = E_layers[1], ν)
    air = SetMaterialParams(;
        Phase = air_phase,
        Density = ConstantDensity(; ρ = 1kg / m^3),
        HeatCapacity = ConstantHeatCapacity(; Cp = 1.0e6J / kg / K),
        Conductivity = ConstantConductivity(; k = 15Watt / K / m),
        CompositeRheology = CompositeRheology((LinearViscous(; η = η_air), el_air)),
        Elasticity = el_air,
        Gravity = ConstantGravity(; g),
        CharDim = CD,
    )
    return (layers..., host, magma, air)
end

# Linear geotherm in the rock, `Ttop` in the air; ghost rows are set by the
# thermal boundary conditions.
@parallel_indices (i, j) function init_geotherm!(T, y, Ttop, dTdz)
    T[i + 1, j + 1] = Ttop + dTdz * max(-y[j], zero(y[j]))
    return nothing
end

# Cells (in the ghosted temperature layout) whose centre lies in the chamber.
@parallel_indices (i, j) function chamber_mask!(mask, x, y, xc, yc, a, b)
    mask[i + 1, j + 1] = ((x[i] - xc) / a)^2 + ((y[j] - yc) / b)^2 ≤ 1
    return nothing
end

# Sea-level record as (age [ka], sea level [m]), ages ascending.
function read_sea_level_record(path)
    lines = filter(l -> !startswith(l, "#"), readlines(path))
    header = split(lines[1], ',')
    header[1:2] == ["time_ka", "RSL_Pmax_m"] ||
        error("unexpected columns $(header[1:2]) in $path")
    data = [parse.(Float64, split(l, ',')[1:2]) for l in lines[2:end]]
    age, sea_level = first.(data), last.(data)
    issorted(age) || error("ages in $path are not ascending")
    return age, sea_level
end

function interpolate_sea_level(age, sea_level, a)
    first(age) ≤ a ≤ last(age) ||
        throw(ArgumentError("age $a ka lies outside the record, [$(first(age)), $(last(age))] ka"))
    k = searchsortedlast(age, a)
    k == lastindex(age) && return sea_level[k]
    w = (a - age[k]) / (age[k + 1] - age[k])
    return (1 - w) * sea_level[k] + w * sea_level[k + 1]
end

# Sea level [m] at each load step, the reference level h_ref [m] of the initial
# equilibrium, the step length [kyr], and the age [ka] of each step (`nothing`
# for the synthetic history).
function sea_level_history(history; age_start = 240.0, age_end = 0.0, dt_kyr = 0.5)
    if history == :steps
        return collect(-10.0:-10.0:-110.0), 0.0, 1.0, nothing
    elseif history == :Grant2014
        age_start > age_end || throw(ArgumentError("age_start must exceed age_end"))
        age, sea_level = read_sea_level_record(joinpath(@__DIR__, "Grant2014_RedSea_RSL.csv"))
        ages = collect(age_start:(-dt_kyr):age_end)
        h = [interpolate_sea_level(age, sea_level, a) for a in ages]
        return h[2:end], h[1], dt_kyr, ages[2:end]
    else
        throw(ArgumentError("unknown sea-level history `$history`; use `:steps` or `:Grant2014`"))
    end
end

# Index of the phase with the largest volume fraction in each cell, on the device.
@parallel_indices (I...) function dominant_phase!(phase, ratios)
    k_max = 1
    for k in cellaxes(ratios)
        if @index(ratios[k, I...]) > @index(ratios[k_max, I...])
            k_max = k
        end
    end
    phase[I...] = k_max
    return nothing
end

function dominant_phase(phase_ratios, ni)
    phase = @zeros(ni...)
    @parallel (@idx ni) dominant_phase!(phase, phase_ratios.center)
    return round.(Int, Array(phase))
end

@parallel_indices (i, j) function rock_buoyancy!(ρg_rock, ρg, rock_fraction)
    ρg_rock[i, j] = ρg[i, j] * rock_fraction[i, j]
    return nothing
end

# The variational solver imposes zero normal traction at the *top face* of the cell
# the marker chain cuts, so the pressure it holds there is `τyy + ρg·dz/2` — the
# pressure of a column whose surface sits at that face, not at the chain. The
# lithostatic initial field has to match: integrating `ρg * ϕ.center` puts half the
# cut cell's own reduced weight in it, and dividing by the rock fraction turns that
# back into `ρg·dz/2`. Starting anywhere else leaves the surface row fighting the
# solver, which a compressible rheology then locks in through `(P - P0)/ηb`.
@parallel_indices (i, j) function _rescale_variational_pressure!(P, rock_fraction)
    ϕ = rock_fraction[i, j]
    P[i, j] = ϕ > 0 ? P[i, j] / ϕ : zero(eltype(P))
    return nothing
end

rescale_variational_pressure!(P, ϕ::JustRelax.RockRatio) =
    @parallel (@idx size(P)) _rescale_variational_pressure!(P, ϕ.center)

# Lithostatic pressure: weight of the laterally averaged rock column plus the
# seawater load `Δp` on its surface. Averaging ρg along x (weighted by the cell
# widths `dx`) gives a reference that is uniform along x, so local density
# anomalies such as the chamber show up in P - P_litho instead of in P_litho.
# Pressure rows in the variational system exclude sticky air, so its overburden
# is excluded as well.
function compute_variational_lithostatic_pressure!(P_litho, ρg_litho, ρg, ϕ::JustRelax.RockRatio, Δp, dx, dz, igg)
    # The x-average below is local to the rank.
    igg.dims[1] == 1 || throw(ArgumentError("the laterally averaged lithostatic pressure needs a single rank along x"))
    @parallel (@idx size(P_litho)) rock_buoyancy!(ρg_litho, ρg, ϕ.center)
    ρg_litho .= sum(ρg_litho .* dx; dims = 1) ./ sum(dx)
    compute_lithostatic_pressure!(P_litho, ρg_litho, dz, igg)
    rescale_variational_pressure!(P_litho, ϕ)
    P_litho .+= Δp .* (ϕ.center .> 0)
    return P_litho
end

# Principal values σ1 ≥ σ2 of the Cauchy stress σ = τ - P I (tension positive).
function principal_stress_values(τxx, τyy, τxy, P)
    σxx = τxx .- P
    σyy = τyy .- P
    a = @. (σxx + σyy) / 2
    b = @. sqrt(((σxx - σyy) / 2)^2 + τxy^2)
    return a .+ b, a .- b
end

# Unit principal directions from a `PrincipalStress` field (eigenvalue times unit
# eigenvector, components along the first axis). Cells with a zero eigenvalue
# have no defined direction and get NaN.
function principal_directions(v)
    n = sqrt.(sum(abs2, v; dims = 1))
    return @views v[1, :, :] ./ n[1, :, :], v[2, :, :] ./ n[1, :, :]
end

# Pore-fluid pressure Pf. Plastic yielding is evaluated at the effective pressure
# P - Pf; momentum and continuity use the total pressure P.
# - `:none`: dry rock, Pf = 0.
# - `:hydrostatic`: seawater column below the surface plus the sea-level load Δp.
# - `:lithostatic`: Pf = P_litho, so the effective pressure vanishes in the
#   lithostatic state.
function update_pore_pressure!(Pf, mode, P_hydro, P_litho, ϕ::JustRelax.RockRatio, Δp)
    if mode == :none
        fill!(Pf, zero(eltype(Pf)))
    elseif mode == :hydrostatic
        Pf .= (P_hydro .+ Δp) .* (ϕ.center .> 0)
    elseif mode == :lithostatic
        Pf .= P_litho
    else
        throw(ArgumentError("unknown pore pressure `$mode`; use `:none`, `:hydrostatic` or `:lithostatic`"))
    end
    return Pf
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny; 
    history = :Grant2014, 
    do_vtk = true, 
    nplot = 1, 
    t_spinup_kyr = 5.0, 
    pore_pressure = :none
)

    # Characteristic scales
    CD = GEO_units(; length = 10km, viscosity = 1.0e21Pa * s, temperature = 1000C)

    # Domain -------------------------------------------------------------------------
    sticky_air = nondimensionalize(2km, CD)    # air layer above the surface
    D = nondimensionalize(20km, CD)            # crustal thickness
    lx = nondimensionalize(100km, CD)          # model width
    ni = nx, ny

    # Magma chamber and layering
    xc_chamber = 0.0
    yc_chamber = -nondimensionalize(4km, CD)       # chamber centre depth
    a_chamber = nondimensionalize(3km, CD)         # horizontal semi-axis (6 km wide)
    b_chamber = nondimensionalize(0.5km, CD)       # vertical semi-axis (sill-like)
    layer_thickness = nondimensionalize(500m, CD)

    # Refine around the chamber and its roof
    Mx = window_monitor(5.0, 4.0, nondimensionalize(8km, CD), xc_chamber)
    My = window_monitor(3.0, 4.0, nondimensionalize(4km, CD), -nondimensionalize(3km, CD))
    xv_ref = solve_grid(-lx / 2, lx / 2, Mx, nx)
    yv_ref = solve_grid(-D, sticky_air, My, ny)
    grid = Geometry(PTArray(backend), xv_ref, yv_ref)
    (; xci, xvi) = grid
    # Host copies of the coordinates for plotting and host-side comprehensions.
    xci_cpu = Array.(xci)
    xvi_cpu = Array.(xvi)

    # Rheology -----------------------------------------------------------------------
    rheology = init_rheology(
        CD;
        E_layers = (5GPa, 10GPa, 15GPa, 20GPa, 30GPa),
        E_host = 40GPa,
        ν = 0.25,
        ρ_rock = 2700kg / m^3,
        creep_rock = SetDislocationCreep(GeoParams.Dislocation.plagioclase_An75_Ji_1993),
        plasticity = DruckerPrager_regularised(; C = 10MPa, ϕ = 30, η_vp = 1.0e19Pa * s, Ψ = 0),
        k_rock = 3.0Watt / K / m,
        Cp = 1050J / kg / K,
        ρ_magma = 2650kg / m^3,
        η_magma = 1.0e18Pa * s,
        G_magma = 30GPa,
        k_magma = 1.5Watt / K / m,
        η_air = 1.0e17Pa * s,
    )
    cutoff_visc = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CD)

    # Sea-level history: one solve per step
    ρw = 1025kg / m^3
    gw = 9.8m / s^2
    sea_levels, h_ref, dt_kyr, ages = sea_level_history(history)
    sea_level_times = isnothing(ages) ? collect(0:length(sea_levels)) .* dt_kyr : [first(ages) + dt_kyr; ages]
    sea_level_values = [h_ref; sea_levels]
    dt = nondimensionalize(dt_kyr * 1.0e3yr, CD)

    # Particles and phases -----------------------------------------------------------
    nxcell, max_xcell, min_xcell = 30, 40, 15
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, Array.(grid.xi_vel[1]), Array.(grid.xi_vel[2]))
    pPhases, = init_cell_arrays(particles, Val(1))
    init_phases!(pPhases, particles, layer_thickness, xc_chamber, yc_chamber, a_chamber, b_chamber)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Flat surface at sea level. The deformation is infinitesimal, so particles and
    # surface stay fixed and every step is solved on the initial geometry.
    nxcell, max_xcell, min_xcell = 50, 100, 25
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], 0.0)
    ϕ_R = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ_R, chain, xvi, grid.di.vertex)

    # Thermal ------------------------------------------------------------------------
    thermal = ThermalArrays(backend, ni)
    Ttop = nondimensionalize((20 + 273)K, CD)
    dTdz = nondimensionalize(30K / km, CD)
    Tbot = Ttop + dTdz * D
    T_chamber = nondimensionalize((1100 + 273)K, CD)
    @parallel (@idx ni) init_geotherm!(thermal.T, xci[2], Ttop, dTdz)
    chamber = @zeros(size(thermal.T)...)
    @parallel (@idx ni) chamber_mask!(chamber, xci..., xc_chamber, yc_chamber, a_chamber, b_chamber)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
        dirichlet = (; constant = T_chamber, mask = chamber),
    )
    # Initial chamber temperature; the Dirichlet condition holds it during the solve.
    thermal.T .= ifelse.(chamber .> 0, T_chamber, thermal.T)
    thermal_bcs!(thermal, thermal_bc)
    @copy thermal.Told thermal.T

    # Stokes -------------------------------------------------------------------------
    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    Pf = @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt, Pf)
    # Deviatoric principal stresses. σ = τ - P I has the same eigenvectors.
    τ_principal = PrincipalStress(backend, ni)

    # Lithostatic initial pressure
    compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase)
    ρg_litho = @zeros(ni...)
    P_litho = @zeros(ni...)
    compute_variational_lithostatic_pressure!(P_litho, ρg_litho, ρg[2], ϕ_R, 0.0, grid.di.vertex..., igg)
    stokes.P .= P_litho
    P_litho0 = Array(P_litho)
    ρwg = nondimensionalize(ρw * gw, CD)
    P_hydro = PTArray(backend)([ρwg * max(-y, zero(y)) for _ in xci_cpu[1], y in xci_cpu[2]])
    update_pore_pressure!(Pf, pore_pressure, P_hydro, P_litho, ϕ_R, 0.0)

    # Free-slip box. The embedded rock–air interface is the free (loaded) surface
    # and the box top lies in the air.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # Strain-rate guess for the first power-law viscosity evaluation
    stokes.ε.xx .= nondimensionalize(1.0e-15 / s, CD)
    compute_viscosity!(stokes, phase_ratios, ϕ_R, args, rheology, cutoff_visc; air_phase)
    di_min = minimum.(grid.di.vertex)
    li = lx, D + sticky_air
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di_min, li; ϵ = 1.0e-5, CFL = 0.8 / √2.1
    )
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ_R, grid.di, dt; ϵ = 1.0e-6)

    # IO -----------------------------------------------------------------------------
    figdir = "Santorini_SeaLevel"
    take(figdir)
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    x_km = ustrip.(dimensionalize(xci_cpu[1], km, CD))
    depth_km = -ustrip.(dimensionalize(xci_cpu[2], km, CD))
    chamber_outline = let θ = LinRange(0, 2π, 200)
        (
            ustrip(dimensionalize(xc_chamber, km, CD)) .+ ustrip(dimensionalize(a_chamber, km, CD)) .* cos.(θ),
            -ustrip(dimensionalize(yc_chamber, km, CD)) .- ustrip(dimensionalize(b_chamber, km, CD)) .* sin.(θ),
        )
    end

    # Rock cells within `δ_roof` of the chamber wall, where the roof stress is sampled
    δ_roof = nondimensionalize(300m, CD)
    phase_center = dominant_phase(phase_ratios, ni)
    near_chamber = [
        ((x - xc_chamber) / (a_chamber + δ_roof))^2 + ((y - yc_chamber) / (b_chamber + δ_roof))^2 ≤ 1
            for x in xci_cpu[1], y in xci_cpu[2]
    ] .& (phase_center .!= magma_phase)
    any(near_chamber) || error("no rock cells resolve the chamber wall; refine the grid")
    time_kyr = Float64[]
    σ1_roof = Float64[]

    # Thermal spin-up: heat diffuses from the chamber before the first Stokes
    # solve, in steps of the load-history step length.
    n_spinup = t_spinup_kyr / dt_kyr
    isinteger(n_spinup) ||
        throw(ArgumentError("t_spinup_kyr = $t_spinup_kyr is not a multiple of the step length $dt_kyr kyr"))
    for _ in 1:Int(n_spinup)
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (; igg, phase = phase_ratios, iterMax = 10.0e3, nout = 1.0e2, verbose = false),
        )
    end

    # Load steps ---------------------------------------------------------------------
    t = 0.0
    for (it, h) in enumerate(sea_levels)
        Δp_dim = ρw * gw * (h - h_ref) * m       # change of seawater pressure
        Δp = nondimensionalize(Δp_dim, CD)
        compute_variational_lithostatic_pressure!(P_litho, ρg_litho, ρg[2], ϕ_R, Δp, grid.di.vertex..., igg)
        update_pore_pressure!(Pf, pore_pressure, P_hydro, P_litho, ϕ_R, Δp)

        solve_VariationalDYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ_R,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                air_phase,
                surface_pressure = Δp,
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 100.0e3,
                total_iterMax = 100.0e3,
                nout = 50,
                rel_drop = 1.0e-2,
                linear_viscosity = false,
                viscosity_cutoff = cutoff_visc,
                free_surface = true,
            ),
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε_pl)

        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (; igg, phase = phase_ratios, iterMax = 10.0e3, nout = 1.0e2, verbose = false),
        )
        t += dt

        println("\n")
        label = isnothing(ages) ? "" : @sprintf("age %.1f ka, ", ages[it])
        label *= @sprintf("sea level %.1f m", h)
        println(label, @sprintf(", Δp = %.3f MPa", ustrip(uconvert(MPa, Δp_dim))))
        println("\n")

        # Output -------------------------------------------------------------------
        igg.me == 0 || continue
        τxx, τyy, τxy, P = Array(stokes.τ.xx), Array(stokes.τ.yy), Array(stokes.τ.xy_c), Array(stokes.P)
        # Total stress, and its change from the initial lithostatic state (τ = 0,
        # P = P_litho0). Both share the eigenvectors of τ.
        σ1, σ2 = principal_stress_values(τxx, τyy, τxy, P)
        Δσ1, Δσ2 = principal_stress_values(τxx, τyy, τxy, P .- P_litho0)
        overpressure = P .- Array(P_litho)
        σ1_MPa, σ2_MPa, Δσ1_MPa, Δσ2_MPa, Pover_MPa = map((σ1, σ2, Δσ1, Δσ2, overpressure)) do A
            A_MPa = ustrip.(dimensionalize(A, MPa, CD))
            # Air cells carry no stress.
            A_MPa[Array(ϕ_R.center) .== 0] .= NaN
            A_MPa
        end
        push!(time_kyr, it * dt_kyr)
        push!(σ1_roof, maximum(Δσ1_MPa[near_chamber]))
        println("  max Δσ1 at chamber wall = $(round(σ1_roof[end]; digits = 2)) MPa")

        (it == 1 || it % nplot == 0 || it == length(sea_levels)) || continue

        if do_vtk
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            compute_principal_stresses!(stokes, τ_principal)
            e1x, e1y = principal_directions(Array(τ_principal.σ1))
            e2x, e2y = principal_directions(Array(τ_principal.σ2))
            data_v = (;
                τxy = ustrip.(dimensionalize(Array(stokes.τ.xy), MPa, CD)),
                Vx = ustrip.(dimensionalize(Array(Vx_v), cm / yr, CD)),
                Vy = ustrip.(dimensionalize(Array(Vy_v), cm / yr, CD)),
            )
            data_c = (;
                phase = phase_center,
                P = ustrip.(dimensionalize(Array(stokes.P), MPa, CD)),
                Pf = ustrip.(dimensionalize(Array(Pf), MPa, CD)),
                T = ustrip.(dimensionalize(Array(thermal.T)[2:(end - 1), 2:(end - 1)], C, CD)),
                η = ustrip.(dimensionalize(Array(stokes.viscosity.η_vep), Pa * s, CD)),
                η_creep = ustrip.(dimensionalize(Array(stokes.viscosity.η), Pa * s, CD)),
                τxx = ustrip.(dimensionalize(Array(stokes.τ.xx), MPa, CD)),
                τyy = ustrip.(dimensionalize(Array(stokes.τ.yy), MPa, CD)),
                τII = ustrip.(dimensionalize(Array(stokes.τ.II), MPa, CD)),
                εII = ustrip.(dimensionalize(Array(stokes.ε.II), s^-1, CD)),
                εII_pl = ustrip.(dimensionalize(Array(stokes.ε_pl.II), s^-1, CD)),
                # principal stresses of σ = τ - P I: values and vectors
                # (value times unit direction, for glyphs)
                σ1 = σ1_MPa,
                σ2 = σ2_MPa,
                σ1_x = σ1_MPa .* e1x,
                σ1_y = σ1_MPa .* e1y,
                σ2_x = σ2_MPa .* e2x,
                σ2_y = σ2_MPa .* e2y,
                Δσ1 = Δσ1_MPa,
                Δσ2 = Δσ2_MPa,
                P_litho = ustrip.(dimensionalize(Array(P_litho), MPa, CD)),
                overpressure = Pover_MPa,
                ϕ_R = Array(ϕ_R.center),
            )
            velocity_v = (data_v.Vx, data_v.Vy)
            save_vtk(
                joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                xvi_cpu,
                xci_cpu,
                data_v,
                data_c,
                velocity_v,
                t = ustrip(dimensionalize(t, yr, CD)),
            )
        end

        # Tensile stress around the chamber, as in Fig. 2 of Satow et al. (2021),
        # and overpressure
        fig = Figure(; size = (900, 1700))
        axes_kw = (; xlabel = "Distance [km]", ylabel = "Depth [km]", yreversed = true, aspect = DataAspect())
        ax1 = Axis(fig[1, 1]; title = label, axes_kw...)
        ax2 = Axis(fig[2, 1]; axes_kw...)
        hm1 = heatmap!(ax1, x_km, depth_km, Δσ1_MPa; colormap = :turbo, colorrange = (-2, 8))
        contour!(ax1, x_km, depth_km, Δσ1_MPa; levels = [3.5], color = :white, linewidth = 2)
        Pover_max = maximum(abs, filter(isfinite, Pover_MPa); init = 0.0)
        Pover_range = Pover_max > 0 ? (-Pover_max, Pover_max) : (-1.0, 1.0)
        hm2 = heatmap!(ax2, x_km, depth_km, Pover_MPa; colormap = :vik, colorrange = Pover_range)
        for ax in (ax1, ax2)
            lines!(ax, chamber_outline...; color = :black, linewidth = 2)
            limits!(ax, -15, 15, 18, 0)
        end
        Colorbar(fig[1, 2], hm1; label = "Tensile stress due to sea-level change [MPa]")
        Colorbar(fig[2, 2], hm2; label = "Overpressure P - P_litho [MPa]")
        ax_sea = Axis(
            fig[3, 1:2]; xlabel = isnothing(ages) ? "Time [kyr]" : "Age [ka]",
            ylabel = "Sea level [m]", xreversed = !isnothing(ages),
        )
        lines!(ax_sea, sea_level_times, sea_level_values; color = :steelblue, linewidth = 2)
        current_time = sea_level_times[it + 1]
        vlines!(ax_sea, [current_time]; color = (:black, 0.35), linestyle = :dash)
        scatter!(
            ax_sea, [current_time], [h]; marker = :circle, markersize = 16,
            color = :red, strokecolor = :white, strokewidth = 2
        )
        rowsize!(fig.layout, 3, Relative(0.2))
        save(joinpath(figdir, @sprintf("%06d.png", it)), fig)
    end

    # Sea level and change of the first principal stress at the chamber wall through time
    if igg.me == 0
        x = isnothing(ages) ? time_kyr : ages
        xlabel = isnothing(ages) ? "Time [kyr]" : "Age [ka]"
        fig = Figure(; size = (800, 700))
        ax1 = Axis(fig[1, 1]; ylabel = "Sea level [m]", xreversed = !isnothing(ages))
        ax2 = Axis(
            fig[2, 1]; xlabel, ylabel = "Δσ1 at chamber wall [MPa]",
            xreversed = !isnothing(ages),
        )
        hlines!(ax1, [-40]; color = :gray, linestyle = :dash)
        lines!(ax1, x, sea_levels)
        band!(ax2, [extrema(x)...], [3, 3], [4, 4]; color = (:gray, 0.3), label = "Tensile strength")
        lines!(ax2, x, σ1_roof; label = "JustRelax")
        axislegend(ax2; position = :lt)
        linkxaxes!(ax1, ax2)
        save(joinpath(figdir, "roof_stress.png"), fig)
    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

nx, ny = 256, 128
do_vtk = true
history = :Grant2014 # :steps or :Grant2014
# history = :steps     # :steps or :Grant2014
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

pore_pressure = :none # :none, :hydrostatic or :lithostatic
pore_pressure = :lithostatic # :none, :hydrostatic or :lithostatic
main(igg, nx, ny; history, do_vtk, nplot = 10, pore_pressure)
