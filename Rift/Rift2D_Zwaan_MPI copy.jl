# Reference models of Zwaan et al. (2026, JGR Solid Earth, doi:10.1029/2025JB033255):
# the initial rifting phase of a 450 x 200 km continental lithosphere. The paper follows
# it with a tectonic pause and an inversion, which this script does not model. The default
# rifting duration is that of the series A models (Table 1); series B rifts for 15 Myr.
#
# Differences from the paper:
#   - no surface processes (FastScape); the free surface is a marker chain with cut cells,
#   - no viscous strain weakening; plastic strain softens the friction angle only,
#   - Drucker-Prager with a tensile cap, and elasticity, in place of rigid-plastic
#     Drucker-Prager,
#   - the bottom boundary has a prescribed inflow balancing the lateral flux instead of a
#     lithostatic traction,
#   - the vertical grid is refined in a fixed depth window instead of above the 750 °C
#     isotherm.

using CairoMakie
const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

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
using GeoParams

using PoissonGrids

# Load file with all the rheology configurations
include(joinpath(@__DIR__, "Rift2D_Zwaan_rheology.jl"))
include(joinpath(@__DIR__, "Rift2D_Zwaan_setup.jl"))

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Pure-shear field with lateral boundaries moving at ±`v` and the rock column fed through
# the bottom boundary; the bottom inflow balances the lateral outflow of the rock.
function set_plate_velocity!(stokes, flow_bcs, v, lx, xvi, xci, backend)
    ε = 2 * v / lx
    @views stokes.V.Vx[:, 2:(end - 1)] .= PTArray(backend)([ε * x for x in Array(xvi[1]), y in Array(xci[2])])
    @views stokes.V.Vy[2:(end - 1), :] .= PTArray(backend)([-ε * y * (y < 0) for x in Array(xci[1]), y in Array(xvi[2])])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)
    return nothing
end

# copy cell-centered `A` into `A_ghosted`, which has one more cell on every side, and
# fill the ghost cells with their nearest interior values
function fill_ghosted!(A_ghosted, A)
    @views A_ghosted[2:(end - 1), 2:(end - 1)] .= A
    @views A_ghosted[1, :] .= A_ghosted[2, :]
    @views A_ghosted[end, :] .= A_ghosted[end - 1, :]
    @views A_ghosted[:, 1] .= A_ghosted[:, 2]
    @views A_ghosted[:, end] .= A_ghosted[:, end - 1]
    return nothing
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(
        xvi_km, igg; v_plate = 5.0, t_rift = 5.0,
        figdir = "figs2D", do_vtk = false, nout = 10,
        ni = Tuple(JustRelax.ImplicitGlobalGrid.global_grid().nxyz[1:2]), max_steps = typemax(Int), do_figures = igg.nprocs == 1,
        nxcell = 75, max_xcell = 100, min_xcell = 50, solver_kwargs = (;),
    )

    do_figures && igg.nprocs > 1 && throw(ArgumentError("figures require a single rank; use parallel VTK output for MPI runs"))
    nout > 0 || throw(ArgumentError("nout must be positive"))

    thickness = (last(xvi_km[1]) - first(xvi_km[1])) * km
    η0 = 1.0e22 * Pa * s
    CharDim = GEO_units(;
        length = thickness, viscosity = η0, temperature = (ZWAAN_GEOMETRY.T_bottom + 273.0) * K
    )
    # Physical domain ------------------------------------
    xvi_nd = ntuple(i -> nondimensionalize(collect(xvi_km[i]) .* km, CharDim), Val(2))
    grid = Geometry(PTArray(backend), xvi_nd, igg)
    (; xci, xvi, ni) = grid   # nodes at the center and vertices of the cells
    li_nd = grid.li
    origin_nd = grid.origin
    dxi = grid.di.vertex                 # cell widths along each axis
    di_min = map(A -> JustRelax.JustRelax2D.minimum_mpi(minimum(A)), grid.di.vertex)    # smallest cell width along each axis
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim)
    dt = nondimensionalize(10.0e3 * yr, CharDim) # initial timestep
    dtmax = nondimensionalize(10.0e3 * yr, CharDim) # maximum timestep
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # material phase, temperature and accumulated plastic strain; `pΔEII` holds the
    # plastic strain increment of one time step and is not advected
    pPhases, pT, pEII, pΔEII = init_cell_arrays(particles, Val(4))

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, pEII, unwrap(pτ)...)
    particle_args_reduced = (pT, pEII, unwrap(pτ)...)

    # Assign particles phases and the initial plastic strain noise
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, CharDim)
    init_plastic_strain!(pEII, particles, CharDim)
    update_cell_halo!(particles.coords..., particles.index, particle_args...)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # Free surface: marker chain and the cut-cell rock fractions built from it
    air_phase = 5
    surface = DistributedMarkerChain(backend_JP, xvi_nd, igg)
    chain = surface.chain
    update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, dxi, air_phase)
    update_cell_halo!(particles.coords..., particles.index, particle_args...)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    pEII.data[pPhases.data .== air_phase] .= 0.0
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, dxi)
    update_halo!(ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    particle2centroid!(stokes.EII_pl, pEII, particles; ghost_1 = false, ghost_2 = false)
    EII_old = similar(stokes.EII_pl)          # plastic strain before the Stokes solve
    EII_ghosted = @zeros(ni .+ 2...)         # plastic strain with ghost cells, for injection
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop = nondimensionalize((ZWAAN_GEOMETRY.T_surface + 273.0) * K, CharDim)
    Tbot = nondimensionalize((ZWAAN_GEOMETRY.T_bottom + 273.0) * K, CharDim)
    thermal = ThermalArrays(backend, ni)
    init_temperature!(thermal.T, xci, CharDim)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = igg.coords[1] == 0, right = igg.coords[1] == igg.dims[1] - 1, top = false, bot = false),
        constant_value = (
            left = false, right = false,
            top = igg.coords[2] == igg.dims[2] - 1 ? Ttop : false,
            bot = igg.coords[2] == 0 ? Tbot : false,
        ),
    )
    thermal_bcs!(thermal, thermal_bc)
    update_halo!(thermal.T)
    thermal.Told .= thermal.T
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure of the rock column. The density depends
    # on pressure, so iterate; the cut-cell weighting is undone after integration.
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    for _ in 1:10
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.T, P = stokes.P); air_phase = air_phase)
        compute_lithostatic_pressure!(stokes.P, ρg[2] .* ϕ.center, dxi[2], igg)
        @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0)
    end
    @copy stokes.P0 stokes.P

    # Rheology
    args0 = (T = thermal.T, P = stokes.P, dt = Inf)
    # not stated in Zwaan et al.; the usual limits of their ASPECT setups
    viscosity_cutoff = nondimensionalize((1.0e19, 1.0e24) .* (Pa * s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di_min, li_nd; ϵ = 1.0e-5, CFL = 0.98 / √2.1
    )

    # Boundary conditions: lateral boundaries moving at ±v_plate, inflow at the base
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (
            left = igg.coords[1] == 0, right = igg.coords[1] == igg.dims[1] - 1,
            top = igg.coords[2] == igg.dims[2] - 1, bot = igg.coords[2] == 0,
        ),
        free_surface = false,
    )
    v = nondimensionalize(v_plate * mm / yr, CharDim)
    tmax = nondimensionalize(t_rift * 1.0e6 * yr, CharDim)
    set_plate_velocity!(stokes, flow_bcs, v, li_nd[1], xvi, xci, backend)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    vtk_dir = joinpath(figdir, "vtk")
    fig_dir = joinpath(figdir, "fig")
    checkpoint = joinpath(figdir, "checkpoint")
    if igg.me == 0
        do_vtk && mkpath(vtk_dir)
        do_figures && mkpath(fig_dir)
        mkpath(checkpoint)
    end
    JustRelax.MPI.Barrier(igg.comm_cart)
    # ----------------------------------------------------

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    centroid2particle!(pT, thermal.T, particles)
    dt₀ = similar(thermal.T)

    P_lith = @zeros(ni...)

    # dimensional host copy of a nondimensional field
    dim(A, unit) = ustrip.(dimensionalize(Array(A), unit, CharDim))
    # dimensional coordinates for output
    xvi_dim = ntuple(i -> dim(xvi[i], km), Val(2))
    xci_dim = ntuple(i -> dim(xci[i], km), Val(2))
    # nondimensional length per km, to write the marker chain in km
    chain_conversion = nondimensionalize(1.0km, CharDim)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; γfact = 20, ϵ = 1.0e-3)

    # Time loop
    t, it = 0.0, 0
    while t < tmax && it < max_steps

        if it == 1
            P_lith .= stokes.P
        end

        # plastic strain before the solve, to extract this step's increment afterwards
        EII_old .= stokes.EII_pl

        args = (; T = thermal.T, P = stokes.P, dt = Inf)

        # Stokes solver -----------------
        t_stokes = @elapsed begin
            result = solve_VariationalDYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                ϕ,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    air_phase = air_phase,
                    verbose_PH = true,
                    verbose_DR = false,
                    iterMax = 100.0e3,
                    total_iterMax = 100.0e3,
                    nout = 20,
                    rel_drop = 1.0e-1,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1.0e-2,
                    pressure_relaxation = 0.5,
                    viscosity_cutoff = viscosity_cutoff,
                    free_surface = true,
                    solver_kwargs...,
                )
            )
        end

        # add this step's plastic strain increment to the particles
        EII_old .= stokes.EII_pl .- EII_old
        centroid2particle!(pΔEII, EII_old, particles; ghosted = false)
        @. pEII.data += ifelse(isnan(pΔEII.data), 0.0, pΔEII.data)

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        dt = compute_dt(stokes, di_min, dtmax, igg)

        if igg.me == 0
            println("Stokes solver time             ")
            println("   Total time:      $t_stokes s")
            println("           Δt:      $(ustrip(dimensionalize(dt, yr, CharDim)) / 1.0e3) kyrs")
        end
        # ------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

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
                iterMax = 10.0e3,
                nout = 2.0e2,
                verbose = igg.me == 0,
            )
        )
        temperature_extrema = (
            JustRelax.JustRelax2D.minimum_mpi(thermal.T),
            JustRelax.JustRelax2D.maximum_mpi(thermal.T),
        )
        igg.me == 0 && @show temperature_extrema

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        # Populate the ghost cells before interpolating to particles.
        @views dt₀[1, :] .= dt₀[2, :]
        @views dt₀[end, :] .= dt₀[end - 1, :]
        @views dt₀[:, 1] .= dt₀[:, 2]
        @views dt₀[:, end] .= dt₀[:, end - 1]
        update_halo!(dt₀)
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion_centroid!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        update_cell_halo!(particles.coords..., particles.index, particle_args...)
        move_particles!(particles, particle_args)
        # advect the free surface and relabel the particles against it before injection
        advect_markerchain!(surface, stokes, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, dxi, air_phase)
        # check if we need to inject particles
        fill_ghosted!(EII_ghosted, stokes.EII_pl)
        update_halo!(EII_ghosted)
        update_cell_halo!(particles.coords..., particles.index, particle_args...)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (thermal.T, EII_ghosted, stress_fields(stokes, pτ)...),
        )
        update_phases_given_markerchain!(pPhases, chain, particles, origin_nd, dxi, air_phase)

        # update phase ratios and rock fractions
        update_cell_halo!(particles.coords..., particles.index, particle_args...)
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, dxi)
        update_halo!(ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy)

        # interpolate fields from particles to centroids
        pT.data[pPhases.data .== air_phase] .= Ttop
        pEII.data[pPhases.data .== air_phase] .= 0.0
        update_cell_halo!(particles.coords..., particles.index, particle_args...)
        particle2centroid!(thermal.T, pT, particles)
        thermal_bcs!(thermal, thermal_bc)
        update_halo!(thermal.T)
        # the friction softening reads the plastic strain carried by the particles
        particle2centroid!(stokes.EII_pl, pEII, particles; ghost_1 = false, ghost_2 = false)

        update_halo!(stokes.EII_pl)
        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)
        update_halo!(
            stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c,
            stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy
        )

        it += 1
        igg.me == 0 && @show it
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, nout) == 0
            checkpointing_jld2(
                checkpoint, stokes, thermal, t, dt, igg; it = it,
                global_vertices = xvi_nd, dims = igg.dims, coords = igg.coords
            )
            checkpointing_particles(
                checkpoint, particles, igg.me;
                phases = pPhases, phase_ratios = phase_ratios, chain = chain,
                particle_args = particle_args, particle_args_reduced = particle_args_reduced,
                t = t, dt = dt, it = it,
                global_vertices = xvi_nd, dims = igg.dims, coords = igg.coords,
            )

            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    Vx = dim(Vx_v, m / s),
                    Vy = dim(Vy_v, m / s),
                )
                data_c = (;
                    T = dim(thermal.T[2:(end - 1), 2:(end - 1)], C),
                    # Pdyno = dim(stokes.P .- P_lith, Pa),
                    P = dim(stokes.P, Pa),
                    η = dim(η, Pa * s),
                    η_vep = dim(η_vep, Pa * s),
                    τII = dim(stokes.τ.II, Pa),
                    τxx = dim(stokes.τ.xx, Pa),
                    τyy = dim(stokes.τ.yy, Pa),
                    εII = dim(stokes.ε.II, s^-1),
                    εII_pl = dim(stokes.ε_pl.II, s^-1),
                    EII_pl = Array(stokes.EII_pl),
                    ρ = dim(ρg[2], m / s^2 * kg / m^3) ./ 9.81,
                )
                velocity_v = (data_v.Vx, data_v.Vy)
                save_pvtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi_dim,
                    xci_dim,
                    data_v,
                    data_c,
                    velocity_v, igg;
                    t = ustrip(dimensionalize(t, yr, CharDim)),
                    pvd = joinpath(vtk_dir, "Rift2D_Zwaan"),
                )
                if igg.me == 0
                    save_marker_chain(
                        joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                        surface;
                        conversion = chain_conversion,
                        t = ustrip(dimensionalize(t, yr, CharDim)),
                        pvd = joinpath(vtk_dir, "Rift2D_Zwaan_topo"),
                    )
                end
            end

            if do_figures && igg.me == 0
                # blank cells that are mostly air
                rock = Array(ϕ.center)
                mask_air(A) = ifelse.(rock .≥ 0.5, A, NaN)
                chain_x = dim(chain.cell_vertices, km)
                chain_y = dim(chain.h_vertices, km)

                ar = DataAspect()
                fig2 = Figure(size = (1200, 900))
                ax1 = Axis(fig2[1, 1], aspect = ar, title = "τxx [MPa]")
                ax2 = Axis(fig2[2, 1], aspect = ar, title = "T [C]")
                ax3 = Axis(fig2[1, 3], aspect = ar, title = "τII [MPa]")
                ax4 = Axis(fig2[2, 3], aspect = ar, title = "log10(η)")
                ax5 = Axis(fig2[3, 1], aspect = ar, title = "EII_pl")
                ax6 = Axis(fig2[3, 3], aspect = ar, title = "ΔP from P_lith [MPa]")
                Label(fig2[0, :], "t = $(round(ustrip(dimensionalize(t, yr, CharDim)) / 1.0e6; digits = 3)) Myr", fontsize = 30)
                h1 = heatmap!(ax1, xci_dim..., mask_air(dim(stokes.τ.xx, MPa)), colormap = :batlow)
                T_dim = dim(thermal.T[2:(end - 1), 2:(end - 1)], C)
                h2 = heatmap!(ax2, xci_dim..., T_dim, colormap = :romaO, colorrange = (0, ustrip(dimensionalize(Tbot, C, CharDim))))
                h3 = heatmap!(ax3, xci_dim..., mask_air(dim(stokes.τ.II, MPa)), colormap = :batlow)
                h4 = heatmap!(ax4, xci_dim..., mask_air(log10.(dim(η_vep, Pa * s))), colormap = :lipari, colorrange = (19, 24))
                h5 = heatmap!(ax5, xci_dim..., mask_air(Array(stokes.EII_pl)), colormap = :batlow)
                h6 = heatmap!(ax6, xci_dim..., mask_air(dim(stokes.P .- P_lith, MPa)), colormap = :roma)
                for ax in (ax1, ax2, ax3, ax4, ax5, ax6)
                    lines!(ax, chain_x, chain_y, color = :red, linewidth = 1.5)
                end

                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                hidexdecorations!(ax4)
                hideydecorations!(ax3)
                hideydecorations!(ax4)
                hideydecorations!(ax6)

                Colorbar(fig2[1, 2], h1)
                Colorbar(fig2[2, 2], h2)
                Colorbar(fig2[1, 4], h3)
                Colorbar(fig2[2, 4], h4)
                Colorbar(fig2[3, 2], h5)
                Colorbar(fig2[3, 4], h6)
                linkaxes!(ax1, ax2, ax3, ax4, ax5, ax6)
                save(joinpath(fig_dir, "sol_" * lpad("$it", 6, "0") * ".png"), fig2)
            end
        end
        # ------------------------------

    end

    return (; stokes, thermal, particles, chain, surface, grid, it, t, dt)
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
function run_rift(; nx = 112, ny = 64, dims = (0, 0), kwargs...)
    MPI = JustRelax.MPI
    MPI.Initialized() || MPI.Init()
    topology = Tuple(MPI.Dims_create(MPI.Comm_size(MPI.COMM_WORLD), collect(dims)))
    all(d -> ((nx, ny)[d] - 2) % topology[d] == 0, 1:2) ||
        throw(ArgumentError("global cell counts minus two must be divisible by process-grid dimensions $topology"))
    ni = ntuple(d -> ((nx, ny)[d] - 2) ÷ topology[d] + 2, 2)
    igg = IGG(
        init_global_grid(
            ni..., 1; dimx = topology[1], dimy = topology[2],
            init_MPI = false, select_device = isCUDA
        )...
    )
    try
        M = window_monitor(5.0, 0.25, 32.5, -27.5)
        xvi_km = refined_grid_vertices(
            (nx, ny), (ZWAAN_GEOMETRY.lx, ZWAAN_GEOMETRY.depth + 10.0),
            (nothing, M); origin = (-ZWAAN_GEOMETRY.lx / 2, -ZWAAN_GEOMETRY.depth), igg,
        )
        return main(xvi_km, igg; ni, kwargs...)
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

# Run only when executed as a script, so `include` (e.g. from test_MPI.jl) just defines `run_rift`.
run_rift(; figdir = "Rift2D_Zwaan_rift", do_vtk = true, nout = 5)
