const isGPU = true
@static if isGPU
    using CUDA
end
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = @static if isGPU
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if isGPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC, JustPIC._3D
# const backend_JP = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend_JP = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

const backend_JP = @static if isGPU
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend
end

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie

# Load file with all the rheology configurations
include("Subduction3D_rheology.jl")
include("Subduction3D_setup.jl")

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
function main3D(li, origin, phases_GMG, igg; nx = 16, ny = 16, nz = 16, figdir = "figs3D", do_vtk = false)

    # Physical domain ------------------------------------
    ni = nx, ny, nz           # number of cells
    di = @. li / ni           # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 125, 150, 75
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni)
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xci, xvi, di)
    # temperature
    particle_args = pPhases, = init_cell_arrays(particles, Val(1))

    # Assign particles phases anomaly
    phases_device = PTArray(backend_JR)(phases_GMG)
    init_phases!(pPhases, phases_device, particles, xvi)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-3, CFL = 0.95 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    temperature2center!(thermal)
    # ----------------------------------------------------
    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel (@idx ni) init_P!(stokes.P, ρg[3], xci[3])
    # stokes.P        .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[end]).* di[end], dims=3), dims=3), dims=3))
    # Rheology
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
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

    # Time loop
    t, it = 0.0, 0
    while it < 500 # run only for 5 Myrs

        # Stokes solver ----------------
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
        t_stokes = @elapsed begin
            out = solve!(
                stokes,
                pt_stokes,
                di,
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
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di) * 0.8
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)

            if do_vtk
                # velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    T = zeros(ni .+ 1...),
                    phase_vertex = [argmax(p) for p in Array(phase_ratios.center)],
                )
                data_c = (;
                    P = Array(stokes.P),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    εzz = Array(stokes.ε.zz),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    τzz = Array(stokes.τ.zz),
                    η = Array(stokes.viscosity.η),
                    Vx_c = diff(stokes.V.Vx[:, 2:(end - 1), 2:(end - 1)], dims = 1),
                    Vy_c = diff(stokes.V.Vy[2:(end - 1), :, 2:(end - 1)], dims = 2),
                    Vz_c = diff(stokes.V.Vz[2:(end - 1), 2:(end - 1), :], dims = 3),
                    phase_center = [argmax(p) for p in Array(phase_ratios.center)],
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
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
nx, ny, nz = 250, 8, 100
li, origin, phases_GMG, = GMG_only(nx + 1, ny + 1, nz + 1)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

# (Path)/folder where output data and figures are stored
figdir = "Subduction3D"
main3D(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);
