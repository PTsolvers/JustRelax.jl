# Whole-mantle convection in a 2D Cartesian box, periodic in x, free slip at the top (z = 0)
# and at the core-mantle boundary, both held at fixed temperature.
# Single material: diffusion creep + elasticity + regularised Drucker-Prager plasticity.
# Temperature and stress live on the particles; the stress is rotated with the vorticity
# every step.
#
#   julia --project=miniapps --startup-file=no \
#       miniapps/DYREL2D/convection/PeriodicGlobalConvection2D_DYREL.jl

const backend_name = "CPU"

using Pkg; Pkg.activate(".")

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

Pkg.activate("miniapps")
using GeoParams, GLMakie, PoissonGrids

function init_rheology(CharDim)
    diffusion = DiffusionCreep(;
        n = 1.0NoUnits,
        r = 0.0NoUnits,
        p = -3.0NoUnits,
        A = 1.5e8MPa^(-1) * μm^3 * s^(-1),
        E = 370.0kJ / mol,
        V = 3.65e-6m^3 / mol,
    )
    elastic = SetConstantElasticity(; G = 85.0e9Pa, ν = 0.4)
    # plastic = DruckerPrager_regularised(; C = 20.0e6Pa, ϕ = 0.56, η_vp = 1.0e20Pa * s, Ψ = 0.0)
    plastic = DruckerPrager_regularised(; C = 20.0e6Pa, ϕ = 30, η_vp = 1.0e20Pa * s, Ψ = 0.0)

    return (
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = 3.5e-5 / K, β = inv(get_Kb(elastic) * Pa), T0 = 293.15K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / K / m),
            CompositeRheology = CompositeRheology((diffusion, elastic, plastic)),
            Elasticity = elastic,
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
    )
end

# Adiabat of potential temperature Tp and gradient dTdz_ad, joined to Ttop and Tbot by linear
# thermal boundary layers of thickness δ, plus uniform random noise in [0, ΔT].
# Depth z in m, temperatures in K.
function initial_temperature(z; ly, Ttop, Tbot, Tp = 1600.0, dTdz_ad = 0.35e-3, δ = 100.0e3, ΔT = 10.0)
    T_ad = Tp + dTdz_ad * z
    T_top = Ttop + (Tp + dTdz_ad * δ - Ttop) * z / δ
    T_bot = Tbot - (Tbot - Tp - dTdz_ad * (ly - δ)) * (ly - z) / δ
    T = z < δ ? T_top : z > ly - δ ? T_bot : T_ad
    return T + ΔT * rand()
end

function main(igg; nx = 64, ny = 24, ar = 8, nt = 10_000, t_end_Myr = 4.5e3, nplot = 5, do_vtk = true, figdir = "PeriodicGlobalConvection2D")

    CharDim = GEO_units(; length = 2890km, viscosity = 1.0e21Pa * s, temperature = 4273.15K)
    Lc = ustrip(dimensionalize(1.0, m, CharDim))
    Tc = ustrip(dimensionalize(1.0, K, CharDim))
    ηc = ustrip(dimensionalize(1.0, Pa * s, CharDim))
    εc = ustrip(dimensionalize(1.0, s^-1, CharDim))
    σc = ustrip(dimensionalize(1.0, MPa, CharDim))
    Vc = ustrip(dimensionalize(1.0, cm / yr, CharDim))

    # Domain -----------------------------------------
    ly = nondimensionalize(2890km, CharDim)
    lx = ar * ly
    ni = nx, ny
    li = lx, ly
    # Uniform in x; in y refined within δ_ref of the top and bottom. The monitor is 1 + α inside
    # both layers and 1 in between, so their cells are ~(1 + α) times thinner. Each tanh
    # transition (width w) is centred w outside its layer, so the whole layer is refined.
    δ_ref = nondimensionalize(100km, CharDim)
    w = nondimensionalize(20km, CharDim)
    α = 4.0
    M_top = tanh_monitor(α, inv(w), -(δ_ref + w); direction = :right)
    M_bot = tanh_monitor(α, inv(w), -ly + δ_ref + w; direction = :left)
    yv = solve_grid(-ly, 0.0, y -> M_top(y) + M_bot(y) - 1, ny)
    grid = Geometry(PTArray(backend), collect(LinRange(0.0, lx, nx + 1)), yv)
    (; xci, xvi) = grid
    di_min = minimum.(grid.di.vertex)

    # Rheology ---------------------------------------
    rheology = init_rheology(CharDim)
    κ = rheology[1].Conductivity[1].k.val / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)
    dt_diff = 0.5 * min(di_min...)^2 / κ / 2.01
    # first step: the diffusive limit is far too long for the elastic terms
    dt = nondimensionalize(1.0e4yr, CharDim)
    viscosity_cutoff = nondimensionalize((1.0e18Pa * s, 1.0e24Pa * s), CharDim)

    # Particles carry temperature and stress; one phase, so the phase ratios are set once.
    particles = init_particles(backend_JP, 25, 35, 12, grid.xi_vel...; periodic = (true, false))
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    pT, pPhases = init_cell_arrays(particles, Val(2))
    pPhases.data .= 1.0
    pτ = StressParticles(particles)
    particle_args = (pT, unwrap(pτ)...)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # Boundary conditions ----------------------------
    # StokesArrays sizes the momentum residuals from flow_bcs: the periodic seam needs its own row.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = true, bot = true),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    Ttop = nondimensionalize(293.15K, CharDim)
    Tbot = nondimensionalize(4273.15K, CharDim)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = false, right = false, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
        periodic = (left = true, right = true, top = false, bot = false),
    )

    stokes = StokesArrays(backend, ni, flow_bcs)
    thermal = ThermalArrays(backend, ni)

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
    # density depends on pressure: iterate ρg and the lithostatic pressure to a consistent state
    for _ in 1:10
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        compute_lithostatic_pressure!(stokes.P, ρg[2], grid.di.vertex[2], igg)
    end
    stokes.ε.II .= nondimensionalize(1.0e-15 / s, CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di_min, li; ϵ = 1.0e-6, CFL = 0.98 / √2.1
    )
    # the default CFL (0.99) diverges on the refined grid
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-3, CFL = 0.95, γfact = 20)

    take(figdir)
    vtk_dir = joinpath(figdir, "vtk")
    do_vtk && take(vtk_dir)
    xvi_km = Array.(xvi) .* (Lc / 1.0e3)
    xci_km = Array.(xci) .* (Lc / 1.0e3)
    Vx_v, Vy_v = @zeros(ni .+ 1...), @zeros(ni .+ 1...)
    dt₀ = similar(thermal.T)
    t, it = 0.0, 0
    t_end = nondimensionalize(t_end_Myr * Myr, CharDim)
    total_iterMax = 50.0e3

    while it < nt && t < t_end

        out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            anderson_PH = 3,
            verbose_PH = true,
            verbose_DR = false,
            iterMax = 50.0e3,
            total_iterMax,
            nout = 50,
            rel_drop = 0.1,
            λ_relaxation_PH = 1,
            λ_relaxation_DR = 1,
            viscosity_relaxation = 1.0e-2,
            viscosity_cutoff,
        )
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

        # Advection
        advection!(particles, RungeKutta2(), @velocity(stokes), dt; periodic_1 = true)
        move_particles!(particles, particle_args; periodic_1 = true)
        inject_particles!(particles, particle_args, (thermal.T, stress_fields(stokes, pτ)...))
        stress2grid!(stokes, pτ, particles)
        particle2centroid!(thermal.T, pT, particles)
        thermal_bcs!(thermal, thermal_bc)

        it += 1
        t += dt
        t_Myr = ustrip(dimensionalize(t, Myr, CharDim))
        igg.me == 0 && println(
            "it = $it, t = $(round(t_Myr; digits = 3)) Myr",
            # solve_DYREL! leaves its loop at total_iterMax without reporting it
            out.iter > total_iterMax ? "  (STOKES NOT CONVERGED: hit total_iterMax)" : "",
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
                    η_vep = Array(stokes.viscosity.η_vep) .* ηc,
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")), xvi_km, xci_km,
                    data_v, data_c, velocity_v; t = t_Myr, pvd = joinpath(vtk_dir, "PeriodicGlobalConvection2D"),
                )
            end

            x_km, y_km = xci_km
            fig = Figure(size = (1600, 900))
            panels = (
                ("T [°C], t = $(round(t_Myr; digits = 1)) Myr", Array(thermal.T[2:(end - 1), 2:(end - 1)]) .* Tc .- 273.15, :lajolla),
                ("log10 η_vep [Pa s]", log10.(Array(stokes.viscosity.η_vep) .* ηc), :batlow),
                ("log10 εII [1/s]", log10.(Array(stokes.ε.II) .* εc), :batlow),
                ("log10 εII plastic [1/s]", log10.(max.(Array(stokes.ε_pl.II), eps()) .* εc), :batlow),
            )
            for (row, (title, field, cmap)) in enumerate(panels)
                ax = Axis(fig[row, 1]; aspect = DataAspect(), title)
                h = heatmap!(ax, x_km, y_km, field; colormap = cmap)
                Colorbar(fig[row, 2], h)
                row < length(panels) && hidexdecorations!(ax)
            end
            save(joinpath(figdir, "$(lpad(it, 6, "0")).png"), fig)
        end
    end

    return stokes, thermal
end

nx, ny = (256, 96) .÷ 2
figdir = "PeriodicGlobalConvection2D_phi30"
# The global grid is not declared periodic (no `periodx`): the x-periodicity lives in the
# boundary conditions, so this setup runs on a single rank.
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; nx, ny, figdir);
