# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)  #or (CUDA, Float64,  3) or (AMDGPU, Float64, 3)

using JustPIC
using JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CellArrays
using GLMakie

# Load file with all the rheology configurations
include("Shearheating_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main3D(igg; ar = 8, ny = 16, nx = ny * 8, nz = ny * 8, figdir = "figs3D", do_vtk = false)

    # Physical domain ------------------------------------
    lx = 70.0e3                             # domain length in x
    ly = 70.0e3                             # domain length in y
    lz = 40.0e3                             # domain length in y
    ni = nx, ny, nz                       # number of cells
    li = lx, ly, lz                       # domain length in x- and y-
    di = @. li / (nx_g(), ny_g(), nz_g())   # grid step in x- and -y
    origin = 0.0, 0.0, -lz                    # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                             # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; is_TP_Conductivity = false)
    κ = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2   # origin of thermal anomaly
    yc_anomaly = ly / 2   # origin of thermal anomaly
    zc_anomaly = 40.0e3  # origin of thermal anomaly
    r_anomaly = 3.0e3    # radius of perturbation
    init_phases!(pPhases, particles, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[3], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[3], xci[3])

    # Rheology
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.95 / √3.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    ## Compression and not extension - fix this
    εbg = 5.0e-14
    stokes.V.Vx .= PTArray(backend_JR)([ -(x - lx / 2) * εbg for x in xvi[1], _ in 1:(ny + 2), _ in 1:(nz + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([ -(y - ly / 2) * εbg for _ in 1:(nx + 2), y in xvi[2], _ in 1:(nz + 2)])
    stokes.V.Vz .= PTArray(backend_JR)([  (lz - abs(z)) * εbg for _ in 1:(nx + 2), _ in 1:(ny + 2), z in xvi[3]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let # let block to avoid polluting the global namespace
        Zv = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[:]), Zv ./ 1.0e3)
        scatter!(ax2, Array(log10.(η[:])), Z ./ 1.0e3)
        ylims!(ax1, minimum(xvi[3]) ./ 1.0e3, 0)
        ylims!(ax2, minimum(xvi[3]) ./ 1.0e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
    end

    grid2particle!(pT, xvi, thermal.T, particles)

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end
    # Time loop
    t, it = 0.0, 0
    while it < 10
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
        compute_ρg!(ρg[3], phase_ratios, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax = 100.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(thermal.T, pT, xvi, particles)
        temperature2center!(thermal)

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
            di;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # interpolate fields from grid vertices to particles
        grid2particle_flip!(pT, xvi, thermal.T, thermal.Told, particles)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (thermal.T,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(T_buffer),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                    Vz = Array(Vz_v),
                )
                data_c = (;
                    Tc = Array(thermal.Tc),
                    P = Array(stokes.P),
                    η = Array(stokes.viscosity.η_vep),
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

            # Make Makie figure
            slice_j = ny >>> 1
            fig = Figure(size = (1200, 1200), title = "t = $t")
            ar = li[1] / li[3]
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [C]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Shear heating [W/m3]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[3] .* 1.0e-3, Array(thermal.T[:, slice_j, :] .- 273.0), colormap = :batlow)
            # Plot particles phase
            h2 = heatmap!(ax2, xvi[1] .* 1.0e-3, xvi[3] .* 1.0e-3, Array(thermal.shear_heating[:, slice_j, :]), colormap = :batlow)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.ε.II[:, slice_j, :])), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep[:, slice_j, :])), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end

figdir = "3D_Benchmark_Duretz_etal_2014"
do_vtk = false # set to true to generate VTK files for ParaView
n = 32
nx = n
ny = n
nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

main3D(igg; ar = ar, ny = ny, nx = nx, nz = nz, figdir = figdir, do_vtk = do_vtk)
