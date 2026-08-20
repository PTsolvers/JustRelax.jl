# Distributed-memory version of Plume3D.jl.
# Rheology after Cloetingh et al. (2022), "Fingerprinting secondary mantle plumes".
#
# Run with, e.g.
#   mpiexecjl -n 8 julia --project=miniapps miniapps/convection/Plume3D/Plume3D_MPI.jl

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

using JustPIC

const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU
end

using GeoParams, Printf

include("Plume3D_rheology.jl")

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

# Lithostatic pressure from the local column. `z` holds global coordinates, so this is
# the same profile on every rank; a cumulative sum along z would only see one subdomain.
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

## MAIN SCRIPT ----------------------------------------------------------------------

function main3D(igg; ar = 1, nx = 16, ny = 16, nz = 16, figdir = "Plume3D_MPI", do_vtk = false)

    # Physical domain ------------------------------------
    # `li` and `origin` describe the *global* domain; `ni` is the number of cells of
    # this rank, and `Geometry` returns the coordinates of its subdomain.
    lz = 700.0e3                          # domain length in z
    lx = ly = lz * ar                     # domain length in x and y
    ni = nx, ny, nz                       # number of local cells
    li = lx, ly, lz                       # global domain length
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid steps
    origin = 0.0, 0.0, -lz                # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                   # nodes at the center and vertices of the cells
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

    update_cell_halo!(particles.coords..., particle_args...)
    update_cell_halo!(particles.index)
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
    update_halo!(thermal.T)
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    @parallel (@idx ni) init_P!(stokes.P, ρg[end], xci[3])

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
    if igg.me == 0
        do_vtk && take(joinpath(figdir, "vtk"))
        take(figdir)
    end
    vtk_dir = joinpath(figdir, "vtk")
    # ----------------------------------------------------

    T_buffer = thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)]
    centroid2particle!(pT, T_buffer, particles)
    dt₀ = similar(stokes.P)

    # Buffers for the MPI gather. Each rank contributes its subdomain minus the halo,
    # so the assembled arrays are (n - 2) * dims cells wide in each direction.
    nx_v, ny_v, nz_v = (nx - 2) * igg.dims[1], (ny - 2) * igg.dims[2], (nz - 2) * igg.dims[3]
    T_v = zeros(nx_v, ny_v, nz_v)
    P_v = zeros(nx_v, ny_v, nz_v)
    τII_v = zeros(nx_v, ny_v, nz_v)
    εII_v = zeros(nx_v, ny_v, nz_v)
    η_vep_v = zeros(nx_v, ny_v, nz_v)
    phases_v = zeros(nx_v, ny_v, nz_v)
    Vx_g = zeros(nx_v, ny_v, nz_v)
    Vy_g = zeros(nx_v, ny_v, nz_v)
    Vz_g = zeros(nx_v, ny_v, nz_v)

    T_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    P_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    τII_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    εII_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    η_vep_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    phases_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    Vx_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    Vy_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    Vz_nohalo = zeros(nx - 2, ny - 2, nz - 2)

    xci_v = ntuple(i -> LinRange(origin[i], origin[i] + li[i], (i == 1 ? nx_v : i == 2 ? ny_v : nz_v)), Val(3))

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    Vz_v = @zeros(ni .+ 1...)
    Vx_c = @zeros(ni...)
    Vy_c = @zeros(ni...)
    Vz_c = @zeros(ni...)

    # Time loop
    t, it = 0.0, 0
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 5 # run only for 5 Myrs

        # interpolate fields from particles to centroids
        particle2centroid!(T_buffer, pT, particles)
        @views thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        update_halo!(thermal.T)
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
        if igg.me == 0
            println("Stokes solver time             ")
            println("   Total time:      $t_stokes s")
            println("   Time/iteration:  $(t_stokes / out.iter) s")
        end
        tensor_invariant!(stokes.ε)
        # `igg` makes the velocity maximum a global reduction, so every rank marches
        # with the same dt
        dt = compute_dt(stokes, di, dt_diff, igg) * 0.8
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
                verbose = igg.me == 0,
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
        # exchange the particles that crossed a subdomain boundary
        update_cell_halo!(particles.coords..., particle_args...)
        update_cell_halo!(particles.index)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,))
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        it += 1
        t += dt
        igg.me == 0 && @printf(
            "it = %d, t = %.3f Myrs, dt = %.3f kyrs\n",
            it, t / (1.0e6 * 3600 * 24 * 365.25), dt / (1.0e3 * 3600 * 24 * 365.25)
        )

        # Data I/O ---------------------
        if do_vtk && (it == 1 || rem(it, 10) == 0)
            velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
            vertex2center!(Vx_c, Vx_v)
            vertex2center!(Vy_c, Vy_v)
            vertex2center!(Vz_c, Vz_v)
            phase_center = [argmax(p) for p in Array(phase_ratios.center)]

            @views T_nohalo .= Array(T_buffer[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views P_nohalo .= Array(stokes.P[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views τII_nohalo .= Array(stokes.τ.II[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views εII_nohalo .= Array(stokes.ε.II[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views phases_nohalo .= phase_center[2:(end - 1), 2:(end - 1), 2:(end - 1)]
            @views Vx_nohalo .= Array(Vx_c[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views Vy_nohalo .= Array(Vy_c[2:(end - 1), 2:(end - 1), 2:(end - 1)])
            @views Vz_nohalo .= Array(Vz_c[2:(end - 1), 2:(end - 1), 2:(end - 1)])

            gather!(T_nohalo, T_v)
            gather!(P_nohalo, P_v)
            gather!(τII_nohalo, τII_v)
            gather!(εII_nohalo, εII_v)
            gather!(η_vep_nohalo, η_vep_v)
            gather!(phases_nohalo, phases_v)
            gather!(Vx_nohalo, Vx_g)
            gather!(Vy_nohalo, Vy_g)
            gather!(Vz_nohalo, Vz_g)

            if igg.me == 0
                data_c = (;
                    T = T_v,
                    P = P_v,
                    τII = τII_v,
                    εII = εII_v,
                    η = log10.(η_vep_v),
                    phase = phases_v,
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xci_v,
                    data_c,
                    (Vx_g, Vy_g, Vz_g);
                    t = t
                )
            end
        end
        # ------------------------------

    end

    finalize_global_grid()
    return nothing
end

## SCRIPT ENTRY POINT ---------------------------------------------------------------

do_vtk = true  # set to true to generate VTK files for ParaView
ar = 1         # aspect ratio
n = 32         # number of cells per direction and per rank
nx = ny = nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

figdir = "Plume3D_MPI_$(nx_g())x$(ny_g())x$(nz_g())"
main3D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk)
