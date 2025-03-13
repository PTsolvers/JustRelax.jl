# using CUDA
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO
const backend_JR = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend_JR = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
# @init_parallel_stencil(CUDA, Float64, 3)
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D
# const backend_JP = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend_JP = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CairoMakie

# Load file with all the rheology configurations
include("Subduction3D_rheology.jl")
include("Subduction3D_setup_MPI.jl")

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
function main3D(x_global, y_global, z_global, li, origin, phases_GMG, igg; nx = 16, ny = 16, nz = 16, figdir = "figs3D", do_vtk = false)
    # LOCAL Physical domain ------------------------------------
    ni = nx, ny, nz           # number of cells
    di = @. li / (nx_g(), ny_g(), nz_g())           # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 150, 175, 125
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni)
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xci, xvi, di)
    # temperature
    particle_args = pPhases, = init_cell_arrays(particles, Val(1))

    # Assign particles phases anomaly
    phases_device = PTArray(backend_JR)(phases_GMG)
    init_phases!(pPhases, phases_device, particles, xvi)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...)
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 5.0e-3, CFL = 0.99 / √3.1)
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
        free_slip = (left = true, right = true, top = true, bot = false, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = true, front = false, back = false),
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

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
        Vx = @zeros(ni...)
        Vy = @zeros(ni...)
        Vz = @zeros(ni...)
    end


    # global array
    nx_v = (nx - 2) * igg.dims[1]
    ny_v = (ny - 2) * igg.dims[2]
    nz_v = (nz - 2) * igg.dims[3]
    # center
    P_v = zeros(nx_v, ny_v, nz_v)
    τII_v = zeros(nx_v, ny_v, nz_v)
    η_vep_v = zeros(nx_v, ny_v, nz_v)
    εII_v = zeros(nx_v, ny_v, nz_v)
    phases_c_v = zeros(nx_v, ny_v, nz_v)
    #center nohalo
    P_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    τII_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    η_vep_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    εII_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    phases_c_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    #vertex
    Vxv_v = zeros(nx_v, ny_v, nz_v)
    Vyv_v = zeros(nx_v, ny_v, nz_v)
    Vzv_v = zeros(nx_v, ny_v, nz_v)
    T_v = zeros(nx_v, ny_v, nz_v)
    #vertex nohalo
    Vxv_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    Vyv_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    Vzv_nohalo = zeros(nx - 2, ny - 2, nz - 2)
    T_nohalo = zeros(nx - 2, ny - 2, nz - 2)

    xci_v = LinRange(minimum(x_global) .* 1.0e3, maximum(x_global) .* 1.0e3, nx_v), LinRange(minimum(y_global) .* 1.0e3, maximum(y_global) .* 1.0e3, ny_v), LinRange(minimum(z_global) .* 1.0e3, maximum(z_global) .* 1.0e3, nz_v)

    # Time loop
    t, it = 0.0, 0

    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 10 # run only for 5 Myrs

        # Stokes solver ----------------
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
                    iterMax = 10.0e3,
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
        dt = compute_dt(stokes, di)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)

        update_cell_halo!(particles.coords..., particle_args...)
        update_cell_halo!(particles.index)

        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        if igg.me == 0
            @show it += 1
            t += dt
        end

        #MPI gathering
        phase_center = [argmax(p) for p in Array(phase_ratios.center)]
        #centers
        @views P_nohalo .= Array(stokes.P[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views τII_nohalo .= Array(stokes.τ.II[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1), 2:(end - 1)])       # Copy data to CPU removing the halo
        @views εII_nohalo .= Array(stokes.ε.II[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views phases_c_nohalo .= Array(phase_center[2:(end - 1), 2:(end - 1), 2:(end - 1)])
        gather!(P_nohalo, P_v)
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)
        gather!(phases_c_nohalo, phases_c_v)
        #vertices
        if do_vtk
            velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
            vertex2center!(Vx, Vx_v)
            vertex2center!(Vy, Vy_v)
            vertex2center!(Vz, Vz_v)
            @views Vxv_nohalo .= Array(Vx[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            @views Vyv_nohalo .= Array(Vy[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            @views Vzv_nohalo .= Array(Vz[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
            gather!(Vxv_nohalo, Vxv_v)
            gather!(Vyv_nohalo, Vyv_v)
            gather!(Vzv_nohalo, Vzv_v)
        end
        @views T_nohalo .= Array(thermal.Tc[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        # Data I/O and plotting ---------------------
        if igg.me == 0 && (it == 1 || rem(it, 1) == 0)
            # checkpointing(figdir, stokes, thermal.T, η, t)
            if do_vtk

                data_c = (;
                    T = T_v,
                    P = P_v,
                    τII = τII_v,
                    εII = εII_v,
                    η = η_vep_v,
                    phases = phases_c_v,


                )
                velocity = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                    Array(Vzv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
                    xci_v ./ 1.0e3,
                    data_c,
                    velocity;
                    t = t / (1.0e6 * 3600 * 24 * 365.25)
                )
            end
        end
        # ------------------------------

    end
    finalize_global_grid()
    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
nx, ny, nz = 32, 32, 32
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

# GLOBAL Physical domain ------------------------------------
x_global = range(-3960, 500, nx_g());
y_global = range(0, 2640, ny_g());
air_thickness = 0.0
z_global = range(-660, air_thickness, nz_g());
origin = (x_global[1], y_global[1], z_global[1])
li = (abs(last(x_global) - first(x_global)), abs(last(y_global) - first(y_global)), abs(last(z_global) - first(z_global)))

ni = nx, ny, nz           # number of cells
di = @. li / (nx_g(), ny_g(), nz_g())           # grid steps
grid_global = Geometry(ni, li; origin = origin)
li_GMG, origin_GMG, phases_GMG, = GMG_only(grid_global.xvi, nx + 1, ny + 1, nz + 1)
# ----------------------------------------------------
# (Path)/folder where output data and figures are stored
figdir = "Subduction3D_$(nx_g())x$(ny_g())x$(nz_g())"

main3D(x_global, y_global, z_global, li_GMG, origin_GMG, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);
