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
include("Subduction3D_setup.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main3D(li, origin, phases_GMG, igg; nx=16, ny=16, nz=16, figdir="figs3D", do_vtk =false)

    # Physical domain ------------------------------------
    ni           = nx, ny, nz           # number of cells
    di           = @. li / ni           # grid steps
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies()
    dt           = 10e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 50, 75, 25
    particles                    = init_particles(backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni)
    subgrid_arrays               = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy, grid_vz    = velocity_grids(xci, xvi, di)
    # temperature
    particle_args = pPhases,     = init_cell_arrays(particles, Val(1))

    # Assign particles phases anomaly
    phases_device = PTArray(backend_JR)(phases_GMG)
    init_phases!(pPhases, phases_device, particles, xvi)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    update_cell_halo!(particles.coords..., particle_args...);
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend_JR, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=5e-3,  CFL = 0.99 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend_JR, ni)
    temperature2center!(thermal)
    # ----------------------------------------------------
    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    @parallel (@idx ni) init_P!(stokes.P, ρg[3], xci[3])
    # stokes.P        .= PTArray(backend_JR)(reverse(cumsum(reverse((ρg[end]).* di[end], dims=3), dims=3), dims=3))
    # Rheology
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1e18, 1e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Boundary conditions
    flow_bcs         = VelocityBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = false, front = true , back = true ),
        no_slip      = (left = false, right = false, top = false, bot = true,  front = false, back = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
        Vz_v = @zeros(ni.+1...)
    end


    # global array
    nx_v         = (nx - 2) * igg.dims[1]
    ny_v         = (ny - 2) * igg.dims[2]
    nz_v         = (nz - 2) * igg.dims[3]
    # center
    P_v          = zeros(nx_v, ny_v, nz_v)
    τII_v        = zeros(nx_v, ny_v, nz_v)
    η_vep_v      = zeros(nx_v, ny_v, nz_v)
    εII_v        = zeros(nx_v, ny_v, nz_v)
    phases_c_v    = zeros(nx_v, ny_v, nz_v)
    #center nohalo
    P_nohalo     = zeros(nx-2, ny-2, nz-2)
    τII_nohalo   = zeros(nx-2, ny-2, nz-2)
    η_vep_nohalo = zeros(nx-2, ny-2, nz-2)
    εII_nohalo   = zeros(nx-2, ny-2, nz-2)
    phases_c_nohalo = zeros(nx-2, ny-2, nz-2)
    #vertex
    Vxv_v        = zeros(nx_v+1, ny_v+1, nz_v+1)
    Vyv_v        = zeros(nx_v+1, ny_v+1, nz_v+1)
    Vzv_v        = zeros(nx_v+1, ny_v+1, nz_v+1)
    T_v          = zeros(nx_v+1, ny_v+1, nz_v+1)
    phases_v_v    = zeros(nx_v+1, ny_v+1, nz_v+1)
    #vertex nohalo
    Vxv_nohalo   = zeros(nx-1, ny-1, nz-1)
    Vyv_nohalo   = zeros(nx-1, ny-1, nz-1)
    Vzv_nohalo   = zeros(nx-1, ny-1, nz-1)
    T_nohalo     = zeros(nx-1, ny-1, nz-1)
    phase_v_nohalo = zeros(nx-1, ny-1, nz-1)

    xci_v        = LinRange(0, 1, nx_v), LinRange(0, 1, ny_v), LinRange(0, 1, nz_v)
    xvi_v        = LinRange(0, 1, nx_v+1), LinRange(0, 1, ny_v+1), LinRange(0, 1, nz_v+1)


    # Time loop
    t, it = 0.0, 0
    # while it < 10000 # run only for 5 Myrs

    while (t/(1e6 * 3600 * 24 *365.25)) < 10 # run only for 5 Myrs

        # # interpolate fields from particle to grid vertices
        # particle2grid!(thermal.T, pT, xvi, particles)
        # temperature2center!(thermal)

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
                kwargs =(;
                    iterMax          = 100e3,
                    nout             = 1e3,
                    viscosity_cutoff = viscosity_cutoff
                )
            );
        end
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt   = compute_dt(stokes, di)
        # ------------------------------

        # # Thermal solver ---------------
        # heatdiffusion_PT!(
        #     thermal,
        #     pt_thermal,
        #     thermal_bc,
        #     rheology,
        #     args,
        #     dt,
        #     di;
        #     igg     = igg,
        #     phase   = phase_ratios,
        #     iterMax = 10e3,
        #     nout    = 1e2,
        #     verbose = true,
        # )
        # subgrid_characteristic_time!(
        #     subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        # )
        # centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        # subgrid_diffusion!(
        #     pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        # )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)

        update_cell_halo!(particles.coords..., particle_args...);
        update_cell_halo!(particles.index)

        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if igg.me == 0 && (it == 1 || rem(it, 20) == 0)
            # checkpointing(figdir, stokes, thermal.T, η, t)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                phase_vertex = [argmax(p) for p in Array(phase_ratios.vertex)]
                phase_center = [argmax(p) for p in Array(phase_ratios.center)]

                # MPI
                #centers
                @views P_nohalo     .= Array(stokes.P[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views τII_nohalo   .= Array(stokes.τ.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:end-1, 2:end-1, 2:end-1])       # Copy data to CPU removing the halo
                @views εII_nohalo   .= Array(stokes.ε.II[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views phases_c_v   .= Array(phase_center[2:end-1, 2:end-1, 2:end-1])
                gather!(P_nohalo, P_v)
                gather!(τII_nohalo, τII_v)
                gather!(η_vep_nohalo, η_vep_v)
                gather!(εII_nohalo, εII_v)
                gather!(phases_c_nohalo, phases_c_v)
                #vertices
                @views Vxv_nohalo   .= Array(Vx_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views Vyv_nohalo   .= Array(Vy_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views Vzv_nohalo   .= Array(Vz_v[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views T_nohalo     .= Array(thermal.T[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
                @views phase_v_nohalo .= Array(phase_vertex[2:end-1, 2:end-1, 2:end-1])
                gather!(Vxv_nohalo, Vxv_v)
                gather!(Vyv_nohalo, Vyv_v)
                gather!(Vzv_nohalo, Vzv_v)
                gather!(T_nohalo, T_v)
                gather!(phase_v_nohalo, phases_v_v)






                data_v = (;
                    T = T_v,
                    phases_v = phases_v_v,
                )
                data_c = (;
                    P = P_v,
                    τII = τII_v,
                    εII = εII_v,
                    η   = η_vep_v,
                    phases = phases_c_v


                )
                velocity_v = (
                    Array(Vxv_v),
                    Array(Vyv_v),
                    Array(Vzv_v),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi_v,
                    xci_v,
                    data_v,
                    data_c,
                    velocity_v
                )
            end
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
# nx,ny,nz = 50, 50, 50
nx,ny,nz = 32,32,32
# nx,ny,nz = 128, 32, 64
li, origin, phases_GMG, = GMG_only(nx+1, ny+1, nz+1)
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
else
    igg
end

# (Path)/folder where output data and figures are stored
figdir   = "Subduction3D_$(nx)x$(ny)x$(nz)"
main3D(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);