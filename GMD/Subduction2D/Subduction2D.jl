const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDA.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
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
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "miniapps")))
using GeoParams, CairoMakie, Dates, Statistics


# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("Subduction2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function custom_argmax(p)
    return p[1] != 0.0 ? 0 : argmax(p)
end

## Publication-ready 3-panel figure ------------------------------------------------
function plot_publication_figure(
    it, t_dim, xvi_dim, xci_dim, thermal, stokes, CharDim, Vx_c, Vy_c, phase_ratios, air_phase, figdir
)
    # Extract data
    T_data = ustrip(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), K, CharDim))
    τII_data = ustrip(dimensionalize(Array(stokes.τ.II), MPa, CharDim))
    η_eff = log10.(ustrip(dimensionalize(Array(stokes.viscosity.η_vep), Pa * s, CharDim)))
    vx_data = ustrip(dimensionalize(Array(Vx_c), cm / yr, CharDim))
    vy_data = ustrip(dimensionalize(Array(Vy_c), cm / yr, CharDim))

    # Publication layout: one aligned panel row and one colorbar row.
    fig = Figure(size = (1800, 260), figure_padding = (8, 12, 6, 8))
    axis_kwargs = (
        aspect = DataAspect(),
        xlabel = "x [km]",
        xticklabelsize = 13,
        yticklabelsize = 13,
        xlabelsize = 15,
        ylabelsize = 15,
        titlesize = 17,
        titlefont = :bold,
        xgridvisible = false,
        ygridvisible = false,
        topspinevisible = true,
        rightspinevisible = true,
    )
    x_rock = (first(xvi_dim[1]), min(3000.0, last(xvi_dim[1])))
    z_rock = (first(xvi_dim[2]), 0.0)
    velocity_to_length = 40.0 # plotted km per cm/yr

    # Panel (a): Temperature with velocity arrows
    ax_a = Axis(fig[1, 1]; axis_kwargs..., ylabel = "z [km]", title = "(a) Temperature")
    h_a = heatmap!(
        ax_a, xci_dim[1], xci_dim[2], T_data,
        colormap = :lajolla, colorrange = (273.0, 1623.0)
    )
    # Velocity is in cm/yr while the axes are in km; preserve magnitude with a
    # fixed visual conversion shared by the field and the reference arrow.
    skip_v = 24
    vx_sample = vx_data[1:skip_v:end, 1:skip_v:end]
    vy_sample = vy_data[1:skip_v:end, 1:skip_v:end]
    arrows2d!(
        ax_a, xci_dim[1][1:skip_v:end], xci_dim[2][1:skip_v:end],
        vx_sample .* velocity_to_length, vy_sample .* velocity_to_length,
        lengthscale = 1, shaftcolor = :black, tipcolor = :black,
        shaftwidth = 4, tipwidth = 12, tiplength = 16
    )
    x_ref = first(x_rock) + 0.08 * (last(x_rock) - first(x_rock))
    z_ref = first(z_rock) + 0.10 * (last(z_rock) - first(z_rock))
    arrows2d!(
        ax_a, [x_ref], [z_ref], [velocity_to_length], [0.0],
        lengthscale = 1, shaftcolor = :black, tipcolor = :black,
        shaftwidth = 4, tipwidth = 12, tiplength = 16
    )
    text!(
        ax_a, x_ref + velocity_to_length / 2, z_ref + 30;
        text = "1 cm/yr", align = (:center, :bottom), color = :black, fontsize = 13,
        font = :bold
    )
    xlims!(ax_a, x_rock)
    ylims!(ax_a, z_rock)
    Colorbar(
        fig[2, 1], h_a;
        vertical = false, label = "Temperature [K]", labelsize = 14, ticklabelsize = 12,
        height = 14, tellheight = true
    )

    # Panel (b): Stress with 1350 K isotherm
    ax_b = Axis(fig[1, 2]; axis_kwargs..., title = "(b) Stress invariant")
    h_b = heatmap!(ax_b, xci_dim[1], xci_dim[2], τII_data, colormap = :lipari)
    contour!(ax_b, xci_dim[1], xci_dim[2], T_data, levels = [1350.0], color = :black, linewidth = 4)
    contour!(ax_b, xci_dim[1], xci_dim[2], T_data, levels = [1350.0], color = :ivory, linewidth = 2.5)
    xlims!(ax_b, x_rock)
    ylims!(ax_b, z_rock)
    hideydecorations!(ax_b, grid = false)
    Colorbar(
        fig[2, 2], h_b;
        vertical = false, label = "τII [MPa]", labelsize = 14, ticklabelsize = 12,
        height = 14, tellheight = true
    )

    # Panel (c): Effective viscosity with 1350 K isotherm
    ax_c = Axis(fig[1, 3]; axis_kwargs..., title = "(c) Effective viscosity")
    h_c = heatmap!(ax_c, xci_dim[1], xci_dim[2], η_eff, colormap = :bilbao)
    contour!(ax_c, xci_dim[1], xci_dim[2], T_data, levels = [1350.0], color = :black, linewidth = 4)
    contour!(ax_c, xci_dim[1], xci_dim[2], T_data, levels = [1350.0], color = :ivory, linewidth = 2.5)
    xlims!(ax_c, x_rock)
    ylims!(ax_c, z_rock)
    hideydecorations!(ax_c, grid = false)
    Colorbar(
        fig[2, 3], h_c;
        vertical = false, label = "log₁₀(η [Pa s])", labelsize = 14, ticklabelsize = 12,
        height = 14, tellheight = true
    )

    Label(
        fig[1, 4], "t = $(round(t_dim; digits = 2)) Myr";
        rotation = pi / 2, fontsize = 16, font = :bold, tellwidth = true
    )
    colsize!(fig.layout, 4, 30)

    # Link axes for consistency
    linkaxes!(ax_a, ax_b, ax_c)

    return fig
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false, abstol = 1.0e-6, γfact = 50.0)

    thickness = 710.0e3 * m
    η0 = 1.0e22 * Pa * s
    CharDim = GEO_units(;
        length = thickness, viscosity = η0, temperature = (1.35e3 + 273)K
    )
    li_nd = nondimensionalize(li .* m, CharDim)
    origin_nd = nondimensionalize(origin .* m, CharDim)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li_nd / ni       # grid steps
    grid = Geometry(ni, li_nd; origin = origin_nd)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim; plastic = true, linear = false)
    rheology_init = init_rheologies(CharDim; plastic = false, linear = true)
    dt = nondimensionalize(10.0e3 * yr, CharDim)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 40
    max_xcell = 60
    min_xcell = 20
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    # pT lives at the cell centers, matching the centered temperature grid
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # velocity grids, needed to advect the marker chain
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # marker chain
    air_phase = 8
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase, particle_args_reduced)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    # thermal.T is cell-centered with one ghost node per boundary. The setup returns
    # the GMG temperature already averaged onto cell centers.
    T_GMG_nd = nondimensionalize(T_GMG * K, CharDim)
    Ttop, Tbot = extrema(T_GMG_nd)
    thermal = ThermalArrays(backend, ni)
    thermal.T[2:(end - 1), 2:(end - 1)] .= PTArray(backend)(T_GMG_nd)

    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, top = false, bot = false),
        constant_value = (; left = false, right = false, top = Ttop, bot = Tbot),
    )
    thermal_bcs!(thermal, thermal_bc)
    thermal.Told .= thermal.T
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))

    # Rheology
    args0 = (T = thermal.T, P = stokes.P, dt = Inf)
    viscosity_cutoff = nondimensionalize((1.0e19, 1.0e24) .* (Pa * s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li_nd; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    dt₀ = similar(stokes.P)
    centroid2particle!(pT, thermal.T, particles)

    # visualization
    xci_dim = ntuple(i -> ustrip(dimensionalize(xci[i], km, CharDim)), Val(2))
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    vtkc = VTKDataSeries(joinpath(figdir, "vtk_series"), xci_dim)

    # DYREL Stokes solver.
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-3, γfact = 100, c_fact = 0.5)

    # Time loop
    t, it = 0.0, 0
    # dt_max = nondimensionalize(100.0e3 * yr, CharDim)
    dt_max = nondimensionalize(10.0e3 * yr, CharDim)
    plot_every = 40

    # thermal relaxation of the GMG initial condition, at fixed geometry
    for _ in 1:100
        args = (; T = thermal.T, P = stokes.P, dt = Inf)
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt)
    end

    dgrain = nondimensionalize((1.0e-3)m, CharDim) # grain size parameter

    # mid-ocean ridge at the left boundary; the layering matches GMG_subduction_2D
    x_ridge = origin_nd[1] + nondimensionalize(50.0e3 * m, CharDim)
    z_crust, z_lith = nondimensionalize((-15.0e3, -95.0e3) .* m, CharDim)

    while it < 5000
        # interpolate temperature from particles to the cell centers
        particle2centroid!(thermal.T, pT, particles; ghost_1 = false, ghost_2 = false)
        thermal_bcs!(thermal , thermal_bc)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        # Stokes solver ----------------
        args = (; T = thermal.T, P = stokes.P, d = dgrain, dt = Inf)
        t_stokes = @elapsed solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            it < 3 ? rheology_init : rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                air_phase = air_phase,
                iterMax = 50.0e3,
                total_iterMax = 50e3,
                nout = 100,
                rel_drop = 1.0e-2,
                λ_relaxation_DR = 1.0,
                λ_relaxation_PH = 1.0,
                viscosity_relaxation = 1.0e-2,
                viscosity_cutoff = viscosity_cutoff,
                free_surface = true,
                verbose_PH = false,
                verbose_DR = false,
            )
        )

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        # compute stress and strain rate 2nd invariants - for plotting
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        tensor_invariant!(stokes.τ)

        # compute time step
        dt = compute_dt(stokes, di, dt_max)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   dt:              $(ustrip(dimensionalize(dt, yr, CharDim)) / 1.0e3) kyrs")

        # Diagnostic: stress levels
        τII_max = maximum(ustrip(dimensionalize(Array(stokes.τ.II), MPa, CharDim)))
        τII_mean = mean(ustrip(dimensionalize(Array(stokes.τ.II), MPa, CharDim)))
        λ_max = maximum(Array(stokes.λ))
        println("   τII max:         $τII_max MPa (mean: $τII_mean MPa)")
        println("   λ max (plastic): $λ_max")
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (
                igg = igg,
                phase = phase_ratios,
                iterMax = 50.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # the ridge is reset first so that the marker chain has the last word on which
        # particles are air
        # reset_ridge!(pPhases, particles, x_ridge, z_crust, z_lith)
        update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, di, air_phase, particle_args_reduced)

        # Inject phase labels first, then initialize every newly injected particle field
        # through the regular centroid/vertex interpolation paths.
        inject_particles_phase!(particles, pPhases, (), ())
        centroid2particle!(pT, thermal.T, particles)
        centroid2particle!(pτ.τ_normal[1], stokes.τ.xx, particles)
        centroid2particle!(pτ.τ_normal[2], stokes.τ.yy, particles)
        grid2particle!(pτ.τ_shear[1], stokes.τ.xy, particles; ghost_1 = false, ghost_2 = false)
        grid2particle!(pτ.ω[1], stokes.ω.xy, particles; ghost_1 = false, ghost_2 = false)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)
        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if do_vtk && (it == 1 || rem(it, plot_every) == 0)
            xvi_dim = ntuple(i -> ustrip(dimensionalize(xvi[i], km, CharDim)), Val(2))
            xci_dim = ntuple(i -> ustrip(dimensionalize(xci[i], km, CharDim)), Val(2))

            pp = [custom_argmax(p) for p in Array(phase_ratios.center)]

            (; η_vep, η) = stokes.viscosity
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            vertex2center!(Vx_c, Vx_v)
            vertex2center!(Vy_c, Vy_v)

            t_dim = ustrip(dimensionalize(t, yr, CharDim) / 1.0e6)
            ρ_real = ustrip(dimensionalize(Array(ρg[2]), kg / m^2 / s^2, CharDim)) ./ 9.81

            data_v = (;
                stress_xy = ustrip(dimensionalize(Array(stokes.τ.xy), Pa, CharDim)),
                strain_rate_xy = ustrip(dimensionalize(Array(stokes.ε.xy), s^-1, CharDim)),
                phase_vertices = [argmax(p) for p in Array(phase_ratios.vertex)],
            )
            data_c = (;
                P = ustrip(dimensionalize(Array(stokes.P), Pa, CharDim)),
                T = ustrip(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), K, CharDim)),
                viscosity_vep = ustrip(dimensionalize(Array(η_vep), Pa * s, CharDim)),
                viscosity = ustrip(dimensionalize(Array(η), Pa * s, CharDim)),
                phases = [argmax(p) for p in Array(phase_ratios.center)],
                EII_pl = Array(stokes.EII_pl),
                stress_II = ustrip(dimensionalize(Array(stokes.τ.II),Pa, CharDim)),
                strain_rate_II = ustrip(dimensionalize(Array(stokes.ε.II), s^-1, CharDim)),
                plastic_strain_rate_II = ustrip(dimensionalize(Array(stokes.ε_pl.II),s^-1, CharDim)),
                density = ustrip(dimensionalize(Array(ρg[2]), kg / m^2 / s^2, CharDim)) ./ 9.81,
            )
            velocity_v = (
                ustrip(dimensionalize(Array(Vx_v), cm / yr, CharDim)),
                ustrip(dimensionalize(Array(Vy_v), cm / yr, CharDim)),
            )
            save_vtk(
                joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                xvi_dim,
                xci_dim,
                data_v,
                data_c,
                velocity_v;
                t = t_dim,
                pvd = joinpath(vtk_dir, "Subduction2D")
                )

            save_marker_chain(
                joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")),
                collect(ustrip(dimensionalize(xvi[1], km, CharDim))),
                collect(ustrip(dimensionalize(Array(chain.h_vertices), km, CharDim)));
                pvd = joinpath(vtk_dir, "Subduction2D_Markerchain"),
                t = t_dim
            )

            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ustrip(dimensionalize(Array(ppx.data[:]), km, CharDim))
            pyv = ustrip(dimensionalize(Array(ppy.data[:]), km, CharDim))
            clr = Array(pPhases.data[:])
            idxv = Array(particles.index.data[:])

            # Make Makie figure
            ar = 3
            fig = Figure(size = (1800, 1200), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[3, 1], aspect = ar, title = "τII [MPa]")
            ax4 = Axis(fig[1, 3], aspect = ar, title = "log10(η) [Pa·s]")
            ax5 = Axis(fig[2, 3], aspect = ar, title = "Vx [cm/yr]")
            ax6 = Axis(fig[3, 3], aspect = ar, title = "Vy [cm/yr]")
            ax7 = Axis(fig[1, 5], aspect = ar, title = "ε̇_pl [s⁻¹]")
            ax8 = Axis(fig[2, 5], aspect = ar, title = "log10(τII) [MPa]")
            ax9 = Axis(fig[3, 5], aspect = ar, title = "log10(ε̇_pl) [s⁻¹]")
            # Plot temperature
            T_data = ustrip(dimensionalize(Array(thermal.T[2:(end - 1), 2:(end - 1)]), K, CharDim))
            h1 = heatmap!(
                ax1, collect(xci_dim[1]), collect(xci_dim[2]), T_data, colormap = :batlow
            )
            contour!(ax1, collect(xci_dim[1]), collect(xci_dim[2]), T_data, levels = 10, color = :white, linewidth = 0.5, alpha = 0.6)
            # Plot particles phase
            h2 = scatter!(ax2, pxv[idxv], pyv[idxv], color = clr[idxv], markersize = 1)
            # Plot 2nd invariant of stress
            h3 = heatmap!(
                ax3, collect(xci_dim[1]), collect(xci_dim[2]), ustrip(dimensionalize(Array((stokes.τ.II)), MPa, CharDim)), colormap = :batlow
            )
            # Overlay marker chain to see stress at air-rock interface
            chain_x = ustrip(dimensionalize(collect(xvi[1]), km, CharDim))
            chain_z = ustrip(dimensionalize(Array(chain.h_vertices), km, CharDim))
            lines!(ax3, chain_x, chain_z, color=:red, linewidth=2, label="Free surface")
            # Plot effective viscosity
            h4 = heatmap!(
                ax4, collect(xci_dim[1]), collect(xci_dim[2]),
                log10.(ustrip(dimensionalize(Array((stokes.viscosity.η_vep)), Pa * s, CharDim))),
                colormap = :batlow
            )
            # Plot horizontal velocity
            vx_data = ustrip(dimensionalize(Array(Vx_c), cm / yr, CharDim))
            vx_max = maximum(abs.(vx_data))
            h5 = heatmap!(
                ax5, collect(xci_dim[1]), collect(xci_dim[2]), vx_data, colormap = :RdBu_11, colorrange = (-vx_max, vx_max)
            )
            # Plot vertical velocity
            vy_data = ustrip(dimensionalize(Array(Vy_c), cm / yr, CharDim))
            vy_max = maximum(abs.(vy_data))
            h6 = heatmap!(
                ax6, collect(xci_dim[1]), collect(xci_dim[2]), vy_data, colormap = :RdBu_11, colorrange = (-vy_max, vy_max)
            )
            # Plot plastic strain rate
            ε_pl_data = ustrip(dimensionalize(Array(stokes.ε_pl.II), s^-1, CharDim))
            h7 = heatmap!(
                ax7, collect(xci_dim[1]), collect(xci_dim[2]), ε_pl_data, colormap = :batlow
            )
            # Plot log10 of stress
            τII_data = ustrip(dimensionalize(Array((stokes.τ.II)), MPa, CharDim))
            τII_safe = max.(τII_data, 1.0e-3)  # avoid log of zero/negative
            h8 = heatmap!(
                ax8, collect(xci_dim[1]), collect(xci_dim[2]), log10.(τII_safe), colormap = :batlow
            )
            # Plot log10 of plastic strain rate
            ε_pl_safe = max.(ε_pl_data, 1.0e-30)  # avoid log of zero/negative
            h9 = heatmap!(
                ax9, collect(xci_dim[1]), collect(xci_dim[2]), log10.(ε_pl_safe), colormap = :batlow
            )
            # Add LAB contour (T = 1300 K) to velocity plots using center-grid temperature
            contour!(ax5, collect(xci_dim[1]), collect(xci_dim[2]), T_data, levels = [1300.0], color = :black, linewidth = 2)
            contour!(ax6, collect(xci_dim[1]), collect(xci_dim[2]), T_data, levels = [1300.0], color = :black, linewidth = 2)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            hidexdecorations!(ax4)
            hidexdecorations!(ax5)
            hidexdecorations!(ax7)
            hidexdecorations!(ax8)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[3, 2], h3)
            Colorbar(fig[1, 4], h4)
            Colorbar(fig[2, 4], h5)
            Colorbar(fig[3, 4], h6)
            Colorbar(fig[1, 6], h7)
            Colorbar(fig[2, 6], h8)
            Colorbar(fig[3, 6], h9)
            linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)
            fig
            save(joinpath(figdir, "$(it).png"), fig)

            # Generate publication-ready 3-panel figure
            fig_pub = plot_publication_figure(
                it, t_dim, xvi_dim, xci_dim, thermal, stokes, CharDim,
                Vx_c, Vy_c, phase_ratios, air_phase, figdir
            )
            save(joinpath(figdir, "publication_$(it).png"), fig_pub)

        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
figdir = "Subduction2D_$(today())"
nx = 1024
ny = 192
# Model setup
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx + 1, ny + 1; v_spread_cm_yr = 3.0, AgeRidge = 0.0, maxAge = 40.0)

; igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(li, origin, phases_GMG, T_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk, abstol = 1.0e-6);
