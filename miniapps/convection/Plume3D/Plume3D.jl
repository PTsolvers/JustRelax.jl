# 3D thermal plume rising through a layered lithosphere.
# Rheology after Cloetingh et al. (2022), "Fingerprinting secondary mantle plumes".

const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")

const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC, JustPIC._3D

const backend_JP = @static if isCUDA
    CUDABackend # Options: JustPIC.CPUBackend, CUDABackend, JustPIC.AMDGPUBackend
else
    JustPIC.CPUBackend
end

using GeoParams, CairoMakie, Printf

## RHEOLOGY -------------------------------------------------------------------------

function init_rheologies()

    # Dislocation and diffusion creep
    disl_upper_crust = DislocationCreep(A = 5.07e-18, n = 2.3, E = 154.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_lower_crust = DislocationCreep(A = 2.08e-23, n = 3.2, E = 238.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_lithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 3.5, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    diff_lithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)
    diff_sublithospheric_mantle = DislocationCreep(A = 2.51e-17, n = 1.0, E = 530.0e3, V = 6.0e-6, r = 0.0, R = 8.3145)

    # Elasticity
    el_upper_crust = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
    el_lower_crust = SetConstantElasticity(; G = 25.0e9, ν = 0.5)
    el_lithospheric_mantle = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
    el_sublithospheric_mantle = SetConstantElasticity(; G = 67.0e9, ν = 0.5)
    β_upper_crust = inv(get_Kb(el_upper_crust))
    β_lower_crust = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle = inv(get_Kb(el_sublithospheric_mantle))

    # Regularised Drucker-Prager plasticity
    η_reg = 1.0e16
    cohesion = 3.0e6
    pl_crust = DruckerPrager_regularised(; C = cohesion, ϕ = asind(0.2), η_vp = η_reg, Ψ = 0.0)
    pl = DruckerPrager_regularised(; C = cohesion, ϕ = asind(0.3), η_vp = η_reg, Ψ = 0.0)

    # Pressure- and temperature-dependent conductivities
    K_crust = TP_Conductivity(; a = 0.64, b = 807.0e0, c = 0.77, d = 0.00004 * 1.0e-6)
    K_mantle = TP_Conductivity(; a = 0.73, b = 1293.0e0, c = 0.77, d = 0.00004 * 1.0e-6)

    return (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.75e3, β = β_upper_crust, T0 = 0.0, α = 3.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity = el_upper_crust,
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.0e3, β = β_lower_crust, T0 = 0.0, α = 3.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Elasticity = el_lower_crust,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_lithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Elasticity = el_lithospheric_mantle,
        ),
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_sublithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity = el_sublithospheric_mantle,
        ),
        # Name              = "Plume",
        SetMaterialParams(;
            Phase = 5,
            Density = PT_Density(; ρ0 = 3.3e3 - 50, β = β_sublithospheric_mantle, T0 = 0.0, α = 3.0e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity = el_sublithospheric_mantle,
        ),
    )
end

## MODEL SETUP ----------------------------------------------------------------------

# Layered lithosphere with a cubic plume of half-width `r` centered at depth `d`
function init_phases!(phases, particles, Lx, Ly; d = 650.0e3, r = 50.0e3)
    ni = size(phases)

    @parallel_indices (I...) function _init_phases!(phases, px, py, pz, index, r, d, Lx, Ly)
        for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            depth = -(@index pz[ip, I...])

            if 0.0e0 ≤ depth ≤ 21.0e3
                @index phases[ip, I...] = 1.0

            elseif 35.0e3 ≥ depth > 21.0e3
                @index phases[ip, I...] = 2.0

            elseif 90.0e3 ≥ depth > 35.0e3
                @index phases[ip, I...] = 3.0

            elseif depth > 90.0e3
                @index phases[ip, I...] = 4.0

            end

            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && ((y - Ly * 0.5)^2 ≤ r^2) && ((depth - d)^2 ≤ r^2)
                @index phases[ip, I...] = 5.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) _init_phases!(phases, particles.coords..., particles.index, r, d, Lx, Ly)
end

# Piecewise-linear geotherm
@parallel_indices (I...) function init_T!(T, z)
    depth = -z[I[3]]

    if depth < 0.0e0
        T[I...] = 273.0

    elseif 0.0e0 ≤ depth < 35.0e3
        dTdZ = (923 - 273) / 35.0e3
        offset = 273.0e0
        T[I...] = depth * dTdZ + offset

    elseif 110.0e3 > depth ≥ 35.0e3
        dTdZ = (1492 - 923) / 75.0e3
        offset = 923.0e0
        T[I...] = (depth - 35.0e3) * dTdZ + offset

    elseif depth ≥ 110.0e3
        dTdZ = (1837 - 1492) / 590.0e3
        offset = 1492.0e0
        T[I...] = (depth - 110.0e3) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _rectangular_perturbation!(T, xc, yc, zc, r, x, y, z)
        if (abs(x[i] - xc) ≤ r) && (abs(y[j] - yc) ≤ r) && (abs(z[k] - zc) ≤ r)
            depth = abs(z[k])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i, j, k] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end

    return @parallel _rectangular_perturbation!(T, xc, yc, zc, r, xvi...)
end

## MAIN SCRIPT ----------------------------------------------------------------------

function main3D(igg; ar = 1, nx = 16, ny = 16, nz = 16, figdir = "Plume3D", do_vtk = false)

    # Physical domain ------------------------------------
    lz = 700.0e3              # domain length in z
    lx = ly = lz * ar         # domain length in x and y
    ni = nx, ny, nz           # number of cells
    li = lx, ly, lz           # domain length
    di = @. li / ni           # grid steps
    origin = 0.0, 0.0, -lz    # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid       # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    κ = 10 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0)
    dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 35, 8
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    subgrid_arrays = SubgridDiffusionCellArrays(particles; loc = :center)
    # temperature and material phase carried by the particles
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Rectangular thermal and compositional anomaly
    xc_anomaly = lx / 2   # origin of thermal anomaly
    yc_anomaly = ly / 2   # origin of thermal anomaly
    zc_anomaly = -610.0e3 # origin of thermal anomaly
    r_anomaly = 50.0e3    # radius of perturbation
    init_phases!(pPhases, particles, lx, ly; d = abs(zc_anomaly), r = r_anomaly)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, Re = 3π, r = 1.0e0, CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )
    # initialize the geotherm on the vertices, then transfer it to the centroids of `thermal.T`
    T_vertex = @zeros(ni .+ 1...)
    @parallel (@idx ni .+ 1) init_T!(T_vertex, xvi[3])
    rectangular_perturbation!(T_vertex, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi)
    vertex2center!(thermal.T, T_vertex; ghost_x = true, ghost_y = true, ghost_z = true)
    thermal_bcs!(thermal, thermal_bc)
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    stokes.P .= PTArray(backend_JR)(reverse(cumsum(reverse(ρg[end] .* di[end], dims = 3), dims = 3), dims = 3))

    # Rheology
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.95 / √3
    )

    # Free slip on every wall
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    T_buffer = thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)]
    centroid2particle!(pT, T_buffer, particles)
    dt₀ = similar(stokes.P)

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end

    # Time loop
    t, it = 0.0, 0
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 5 # run only for 5 Myrs

        # interpolate fields from particles to centroids
        particle2centroid!(T_buffer, pT, particles)
        @views thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        # ------------------------------

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            out = solve!(
                stokes,
                pt_stokes,
                grid,
                flow_bcs,
                ρg,
                phase_ratios,
                rheology,
                args,
                Inf,
                igg;
                kwargs = (;
                    iterMax = 100.0e3,
                    nout = 1.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff) * 0.8
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
            kwargs = (;
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
        subgrid_diffusion_centroid!(
            pT, T_buffer, thermal.ΔT, subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,))
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        it += 1
        t += dt
        @printf("it = %d, t = %.3f Myrs, dt = %.3f kyrs\n", it, t / (1.0e6 * 3600 * 24 * 365.25), dt / (1.0e3 * 3600 * 24 * 365.25))

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(T_vertex),
                )
                data_c = (;
                    T = Array(T_buffer),
                    P = Array(stokes.P),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    η = Array(log10.(stokes.viscosity.η_vep)),
                    phase = [argmax(p) for p in Array(phase_ratios.center)],
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                    Array(Vz_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v,
                    t = t
                )
            end

            # vertical slice through the middle of the plume
            slice_j = ny >>> 1
            fig = Figure(size = (1400, 1200))
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "τII [MPa]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(T_buffer[:, slice_j, :]), colormap = :lajolla)
            h2 = heatmap!(ax2, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(stokes.τ.II[:, slice_j, :] .* 1.0e-6), colormap = :batlow)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.ε.II[:, slice_j, :])), colormap = :batlow)
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep[:, slice_j, :])), colormap = :batlow)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------
    end

    return nothing
end

## SCRIPT ENTRY POINT ---------------------------------------------------------------

do_vtk = true  # set to true to generate VTK files for ParaView
ar = 1         # aspect ratio
n = 32
nx = ny = nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

figdir = "Plume3D_$n"
main3D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk)
