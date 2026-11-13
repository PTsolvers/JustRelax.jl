# Whole-mantle convection in a 2D Cartesian box, periodic in x, under a free surface.
# The free surface is a marker chain with cut cells (`RockRatio`), solved with variational DYREL;
# the air above it is held at the surface temperature. Free slip and fixed temperature at the
# core-mantle boundary.
# Mantle: diffusion creep + elasticity + regularised Drucker-Prager plasticity.
# Temperature, phase and stress live on the particles; the stress is rotated with the vorticity
# every step.
#
#   julia --project=miniapps --startup-file=no \
#       miniapps/DYREL2D/convection/PeriodicGlobalConvection2D_DYREL_VS.jl

const backend_name = "CPU"

using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))

@static if backend_name == "CUDA"
    using CUDA
elseif backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using ParallelStencil

const backend = @static if backend_name == "CUDA"
    JustRelax.CUDABackend
elseif backend_name == "AMDGPU"
    JustRelax.AMDGPUBackend
else
    JustRelax.CPUBackend
end

@static if backend_name == "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
elseif backend_name == "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
const backend_JP = @static if backend_name == "CUDA"
    CUDA.CUDABackend
elseif backend_name == "AMDGPU"
    AMDGPU.ROCBackend
else
    JustPIC.CPU
end

using GeoParams, GLMakie, PoissonGrids

const air_phase = 2

function init_rheology(CharDim; friction_angle, cohesion_softening)
    diffusion = DiffusionCreep(;
        n = 1.0NoUnits,
        r = 0.0NoUnits,
        p = -3.0NoUnits,
        A = 1.5e8MPa^(-1) * μm^3 * s^(-1),
        E = 370.0kJ / mol,
        V = 3.65e-6m^3 / mol,
    )
    elastic = SetConstantElasticity(; G = 85.0e9Pa, ν = 0.4)
    C = 20.0e6Pa
    # cohesion falls linearly from C to C/10 as the accumulated plastic strain grows from 0 to 1
    soft_C = cohesion_softening ? LinearSoftening((C / 10, C), (0.0, 1.0)) : NoSoftening()
    plastic = DruckerPrager_regularised(; C, ϕ = friction_angle, η_vp = 1.0e20Pa * s, Ψ = 0.0, softening_C = soft_C)

    mantle = SetMaterialParams(;
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = 3.5e-5 / K, β = inv(get_Kb(elastic) * Pa), T0 = 293.15K),
        HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
        Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
        CompositeRheology = CompositeRheology((diffusion, elastic, plastic)),
        Elasticity = elastic,
        Gravity = ConstantGravity(; g = 9.81m / s^2),
        CharDim = CharDim,
    )
    air = SetMaterialParams(;
        Phase = air_phase,
        Density = ConstantDensity(; ρ = 1.0e2kg / m^3),
        HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3J / kg / K),
        Conductivity = ConstantConductivity(; k = 1.0Watt / K / m),
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19Pa * s),)),
        Gravity = ConstantGravity(; g = 9.81m / s^2),
        CharDim = CharDim,
    )
    return (mantle, air)
end

# air above the initial surface y = 0, mantle below
function init_phases!(phases, particles)
    ni = size(phases)
    @parallel_indices (i, j) function _init_phases!(phases, py, index)
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue
            @index phases[ip, i, j] = (@index py[ip, i, j]) > 0 ? float(air_phase) : 1.0
        end
        return nothing
    end
    return @parallel (@idx ni) _init_phases!(phases, particles.coords[2], particles.index)
end

# Adiabat of potential temperature Tp and gradient dTdz_ad, joined to Ttop and Tbot by linear
# thermal boundary layers of thickness δ, plus uniform random noise in [0, ΔT]; Ttop in the air.
# Depth z (positive down from the surface) in m, temperatures in K.
function initial_temperature(z; ly, Ttop, Tbot, Tp = 1600.0, dTdz_ad = 0.35e-3, δ = 100.0e3, ΔT = 10.0)
    z < 0 && return Ttop
    T_ad = Tp + dTdz_ad * z
    T_top = Ttop + (Tp + dTdz_ad * δ - Ttop) * z / δ
    T_bot = Tbot - (Tbot - Tp - dTdz_ad * (ly - δ)) * (ly - z) / δ
    T = z < δ ? T_top : z > ly - δ ? T_bot : T_ad
    return T + ΔT * rand()
end

function main(igg; nx = 64, ny = 24, ar = 8, thick_air_km = 50.0, nt = 10_000, t_end_Myr = 4.5e3, nplot = 5, do_vtk = true, figdir = "PeriodicGlobalConvection2D_VS", friction_angle = 30, cohesion_softening = true)
    figdir = string(figdir, "_phi", friction_angle, cohesion_softening ? "_softening" : "_nosoftening")

    CharDim = GEO_units(; length = 2890km, viscosity = 1.0e21Pa * s, temperature = 4273.15K)
    Lc = ustrip(dimensionalize(1.0, m, CharDim))
    Tc = ustrip(dimensionalize(1.0, K, CharDim))
    ηc = ustrip(dimensionalize(1.0, Pa * s, CharDim))
    εc = ustrip(dimensionalize(1.0, s^-1, CharDim))
    σc = ustrip(dimensionalize(1.0, MPa, CharDim))
    Vc = ustrip(dimensionalize(1.0, cm / yr, CharDim))

    # Domain: mantle below y = 0, air above ----------
    ly = nondimensionalize(2890km, CharDim)
    thick_air = nondimensionalize(thick_air_km * km, CharDim)
    lx = ar * ly
    ni = nx, ny
    li = lx, ly + thick_air
    origin = 0.0, -ly
    # Uniform in x; in y refined within δ_ref of the surface and of the bottom. The monitor is
    # 1 + α inside both layers and 1 in between, so their cells are ~(1 + α) times thinner. Each
    # tanh transition (width w) is centred w outside its layer, so the whole layer is refined.
    δ_ref = nondimensionalize(100km, CharDim)
    w = nondimensionalize(20km, CharDim)
    α = 4.0
    M_top = tanh_monitor(α, inv(w), -(δ_ref + w); direction = :right)
    M_bot = tanh_monitor(α, inv(w), -ly + δ_ref + w; direction = :left)
    yv = solve_grid(-ly, thick_air, y -> M_top(y) + M_bot(y) - 1, ny)
    grid = Geometry(PTArray(backend), collect(LinRange(0.0, lx, nx + 1)), yv)
    (; xci, xvi) = grid
    di_min = minimum.(grid.di.vertex)

    # Rheology ---------------------------------------
    rheology = init_rheology(CharDim; friction_angle, cohesion_softening)
    κ = rheology[1].Conductivity[1].k.val / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)
    dt_diff = 0.5 * min(di_min...)^2 / κ / 2.01
    # first step: the diffusive limit is far too long for the elastic and free-surface terms
    dt = nondimensionalize(1.0e4yr, CharDim)
    viscosity_cutoff = nondimensionalize((1.0e18Pa * s, 1.0e24Pa * s), CharDim)

    # Particles: phase, temperature, accumulated plastic strain and stress
    particles = init_particles(backend_JP, 25, 35, 12, grid.xi_vel...; periodic = (true, false))
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    pT, pPhases, pEII = init_cell_arrays(particles, Val(3))
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, pEII, unwrap(pτ)...)
    particle_args_reduced = (pT, pEII, unwrap(pτ)...)
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Free surface: marker chain and cut-cell rock fractions
    chain = init_markerchain(backend_JP, 100, 15, 200, xvi[1], 0.0)
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, grid.di.vertex; periodic_1 = true)

    # Boundary conditions ----------------------------
    # StokesArrays sizes the momentum residuals from flow_bcs: the periodic seam needs its own row.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = true, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    Ttop = nondimensionalize(293.15K, CharDim)
    Tbot = nondimensionalize(4273.15K, CharDim)
    thermal = ThermalArrays(backend, ni)
    # cells entirely in the air are held at Ttop; refreshed from ϕ as the surface moves
    air_mask = @zeros(size(thermal.T)...)
    @views air_mask[2:(end - 1), 2:(end - 1)] .= iszero.(ϕ.center)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = false, right = false, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
        periodic = (left = true, right = true, top = false, bot = false),
        dirichlet = (constant = Ttop, mask = air_mask),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    # the solver softens with, and adds the step's plastic strain to, stokes.EII_pl; the
    # material carries it on the particles, which receive the increment after each solve
    EII0 = similar(stokes.EII_pl)

    # Initial temperature ----------------------------
    thermal.T[2:(end - 1), 2:(end - 1)] .= PTArray(backend)(
        [
            initial_temperature(-y * Lc; ly = ly * Lc, Ttop = Ttop * Tc, Tbot = Tbot * Tc) / Tc
                for _ in Array(xci[1]), y in Array(xci[2])
        ]
    )
    thermal_bcs!(thermal, thermal_bc)
    centroid2particle!(pT, thermal.T, particles)

    # Initial pressure and viscosity -----------------
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    ρg = @zeros(ni...), @zeros(ni...)
    # Density depends on pressure: iterate ρg and the lithostatic pressure to a consistent state.
    # The pressure is that of the rock column, with the cut-cell weighting undone.
    for _ in 1:10
        compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase)
        compute_lithostatic_pressure!(stokes.P, ρg[2] .* ϕ.center, grid.di.vertex[2], igg)
        @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0)
    end
    stokes.ε.II .= nondimensionalize(1.0e-15 / s, CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di_min, li; ϵ = 1.0e-6, CFL = 0.98 / √2.1
    )
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-3, CFL = 0.95, γfact = 20)

    take(figdir)
    vtk_dir = joinpath(figdir, "vtk")
    do_vtk && take(vtk_dir)
    xvi_km = Array.(xvi) .* (Lc / 1.0e3)
    xci_km = Array.(xci) .* (Lc / 1.0e3)
    Vx_v, Vy_v = @zeros(ni .+ 1...), @zeros(ni .+ 1...)
    dt₀ = similar(thermal.T)
    t, it = 0.0, 0
    t_end = nondimensionalize(t_end_Myr * Myr, CharDim)

    while it < nt && t < t_end

        compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase)
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase)
        particle2centroid!(stokes.EII_pl, pEII, particles; ghost_1 = false, ghost_2 = false)
        copyto!(EII0, stokes.EII_pl)
        result = solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            air_phase,
            anderson_PH = 3,
            verbose_PH = true,
            verbose_DR = false,
            iterMax = 50.0e3,
            total_iterMax = 50.0e3,
            nout = 50,
            rel_drop = 0.1,
            λ_relaxation_PH = 1,
            λ_relaxation_DR = 1,
            viscosity_relaxation = 1.0e-2,
            free_surface = true,
            viscosity_cutoff,
        )
        result.converged || error("Variational DYREL did not converge at step $(it + 1); cannot advance the model")
        centroid2particle_flip!(pEII, stokes.EII_pl, EII0, particles; ghosted = false)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dt = compute_dt(stokes, di_min, dt_diff, igg)
        rotate_stress!(pτ, stokes, particles, dt)

        # Thermal diffusion + subgrid diffusion on the particles
        heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
            kwargs = (; igg, phase = phase_ratios, iterMax = 10.0e3, nout = 1.0e1, verbose = false),
        )
        subgrid_characteristic_time!(subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes)
        # Ghost cells of dt₀: periodic in x, mirrored in y
        @views dt₀[1, :] .= dt₀[end - 1, :]
        @views dt₀[end, :] .= dt₀[2, :]
        @views dt₀[:, 1] .= dt₀[:, 2]
        @views dt₀[:, end] .= dt₀[:, end - 1]
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt)

        # Advect the material, then the free surface, then rebuild phases and cut cells
        advection!(particles, RungeKutta2(), @velocity(stokes), dt; periodic_1 = true)
        move_particles!(particles, particle_args; periodic_1 = true)
        semilagrangian_advection_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid.xi_vel, xvi, dt; periodic_1 = true)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, grid.di.vertex, air_phase, particle_args_reduced)
        inject_particles_phase!(particles, pPhases, particle_args_reduced, (thermal.T, stokes.EII_pl, stress_fields(stokes, pτ)...))
        update_phase_ratios!(phase_ratios, particles, pPhases)
        stress2grid!(stokes, pτ, particles)
        compute_rock_fraction!(ϕ, chain, xvi, grid.di.vertex; periodic_1 = true)
        @views air_mask[2:(end - 1), 2:(end - 1)] .= iszero.(ϕ.center)
        particle2centroid!(thermal.T, pT, particles)
        thermal_bcs!(thermal, thermal_bc)

        it += 1
        t += dt
        t_Myr = ustrip(dimensionalize(t, Myr, CharDim))
        igg.me == 0 && println(
            "it = $it, t = $(round(t_Myr; digits = 3)) Myr",
            result.converged ? "" : "  (STOKES NOT CONVERGED, err = $(round(result.err; sigdigits = 3)))",
        )

        if it == 1 || rem(it, nplot) == 0
            if do_vtk
                tensor_invariant!(stokes.τ)
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                velocity_v = (Array(Vx_v) .* Vc, Array(Vy_v) .* Vc)
                data_v = (;
                    τxy = Array(stokes.τ.xy) .* σc,
                    εxy = Array(stokes.ε.xy) .* εc,
                    Vx = velocity_v[1],
                    Vy = velocity_v[2],
                )
                data_c = (;
                    T = Array(thermal.T[2:(end - 1), 2:(end - 1)]) .* Tc .- 273.15,
                    P = Array(stokes.P) .* σc,
                    τII = Array(stokes.τ.II) .* σc,
                    εII = Array(stokes.ε.II) .* εc,
                    εII_pl = Array(stokes.ε_pl.II) .* εc,
                    EII_pl = Array(stokes.EII_pl),
                    η_vep = Array(stokes.viscosity.η_vep) .* ηc,
                    rock_fraction = Array(ϕ.center),
                )
                pvd = joinpath(vtk_dir, "PeriodicGlobalConvection2D_VS")
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")), xvi_km, xci_km,
                    data_v, data_c, velocity_v; t = t_Myr, pvd,
                )
                save_marker_chain(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")), chain; t = t_Myr, pvd = pvd * "_chain")
            end

            x_km, y_km = xci_km
            chain_x = Array(chain.cell_vertices) .* (Lc / 1.0e3)
            chain_y = Array(chain.h_vertices) .* (Lc / 1.0e3)
            # cut cells that are mostly air carry no meaningful mechanical state
            rock = Array(ϕ.center) .≥ 0.5
            fig = Figure(size = (1600, 900))
            panels = (
                ("T [°C], t = $(round(t_Myr; digits = 1)) Myr", Array(thermal.T[2:(end - 1), 2:(end - 1)]) .* Tc .- 273.15, :lajolla),
                ("log10 η_vep [Pa s]", ifelse.(rock, log10.(Array(stokes.viscosity.η_vep) .* ηc), NaN), :batlow),
                ("log10 εII [1/s]", ifelse.(rock, log10.(Array(stokes.ε.II) .* εc), NaN), :batlow),
                ("log10 εII plastic [1/s]", ifelse.(rock, log10.(max.(Array(stokes.ε_pl.II), eps()) .* εc), NaN), :batlow),
            )
            for (row, (title, field, cmap)) in enumerate(panels)
                ax = Axis(fig[row, 1]; aspect = DataAspect(), title)
                h = heatmap!(ax, x_km, y_km, field; colormap = cmap)
                lines!(ax, chain_x, chain_y; color = :black, linewidth = 1)
                Colorbar(fig[row, 2], h)
                row < length(panels) && hidexdecorations!(ax)
            end
            save(joinpath(figdir, "$(lpad(it, 6, "0")).png"), fig)
        end
    end

    return stokes, thermal, chain
end

nx, ny = (256, 96) .÷ 2
figdir = "PeriodicGlobalConvection2D_VS"
friction_angle = 0.57
cohesion_softening = false
# The global grid is not declared periodic (no `periodx`): the x-periodicity lives in the
# boundary conditions, so this setup runs on a single rank.
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; nx, ny, figdir, friction_angle, cohesion_softening);
