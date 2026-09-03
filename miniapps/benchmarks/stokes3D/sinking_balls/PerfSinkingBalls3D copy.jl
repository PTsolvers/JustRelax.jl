# Performance/scaling benchmark of the sinking balls setup of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
#
# Writes the average wall time per pseudo-transient iteration, the particle advection
# time, and the total time per time step of rank 0 to `SinkingTimers_<nprocs>.csv`.
#
# Run with, e.g.
#   mpiexecjl -n 8 julia --project=miniapps miniapps/benchmarks/stokes3D/sinking_balls/PerfSinkingBalls3D.jl

const isCUDA = true
# const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using CSV, DataFrames, GeoParams, Printf

# Load file with all the rheology configurations
include("SinkingBalls_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main3D(igg; nx = 16, ny = 16, nz = 16, figdir = "figs3D", do_vtk = false, nt = 2)

    # Physical domain ------------------------------------
    # `li` and `origin` describe the *global* domain; `ni` is the number of cells of
    # this rank, and `Geometry` returns the coordinates of its subdomain.
    lx = ly = lz = 1.0e0                  # global domain length in x, y and z
    ni = nx, ny, nz                       # number of local cells
    li = lx, ly, lz                       # global domain length
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid steps
    origin = 0.0, 0.0, 0.0                # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid                   # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = dt_diff = 10 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 25, 1
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    # material phase carried by the particles
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)

    # Spherical compositional anomalies
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    update_cell_halo!(particles.coords..., particle_args...)
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, CFL = 0.9 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.T, P = stokes.P))
    @parallel (@idx ni) init_P!(stokes.P, ρg[end], xci[3])

    # Rheology
    viscosity_cutoff = (-Inf, Inf)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Free slip on every wall
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk_$(igg.nprocs)")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end

    # Timers
    timers_file = "SinkingTimers_$(igg.nprocs).csv"

    # Time loop
    t, it = 0.0, 0
    while it < nt
        local iter_time, advection_time
        time2sol = @elapsed begin
            args = (; T = thermal.T, P = stokes.P, dt = Inf)

            # Stokes solver ----------------
            iters = solve!(
                stokes,
                pt_stokes,
                grid,
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
                    viscosity_cutoff = viscosity_cutoff,
                    verbose = igg.me == 0,
                )
            )
            tensor_invariant!(stokes.ε)
            dt = compute_dt(stokes, di, dt_diff)
            iter_time = iters.av_time
            # ------------------------------

            # Advection --------------------
            advection_time = @elapsed begin
                # advect particles in space
                advection!(particles, RungeKutta2(), @velocity(stokes), dt)
                # exchange the particles that crossed a subdomain boundary
                update_cell_halo!(particles.coords..., particle_args...)
                update_cell_halo!(particles.index)
                # advect particles in memory
                move_particles!(particles, particle_args)
            end
            # check if we need to inject particles
            inject_particles_phase!(particles, pPhases, (), ())
            # update phase ratios
            update_phase_ratios!(phase_ratios, particles, pPhases)
            # ------------------------------
        end

        if igg.me == 0
            df = DataFrame(; iter_time, advection_time, time2sol)
            CSV.write(timers_file, df, writeheader = (it == 0), append = true)
        end

        it += 1
        t += dt
        igg.me == 0 && @printf(
            "it = %d, t = %.3e, dt = %.3e, time/iter = %.3e s\n", it, t, dt, iter_time
        )

        # Data I/O ---------------------
        if it == 1 || rem(it, 50) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                )
                data_c = (;
                    P = Array(stokes.P),
                    η = Array(stokes.viscosity.η),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                    Array(Vz_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_rank_$(igg.me)_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v,
                    t = t
                )
            end
        end
        # ------------------------------

    end

    finalize_global_grid()
    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

## SCRIPT ENTRY POINT ---------------------------------------------------------------

figdir = "Perf_SinkingBalls3D"
do_vtk = true # set to true to generate VTK files for ParaView
n = 120       # number of cells per direction and per rank
nx = ny = nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true, select_device = false)...)
else
    igg
end

main3D(igg; nx = nx, ny = ny, nz = nz, figdir = figdir, do_vtk = do_vtk)
