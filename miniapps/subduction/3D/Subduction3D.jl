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
using Printf, LinearAlgebra, GeoParams, GLMakie

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
    nxcell, max_xcell, min_xcell = 40, 60, 20
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
    # Time loop
    t, it = 0.0, 0
    while it < 1000 # run only for 5 Myrs
    # while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs
        
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
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(thermal.T),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                )
                data_c = (;
                    P   = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η   = Array(stokes.viscosity.η),
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
                    velocity_v
                )
            end

            slice_j = ny >>> 1
            # Make Makie figure
            fig = Figure(size = (1400, 1800), title = "t = $t")
            ax1 = Axis(fig[1,1], title = "P [GPA]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], title = "τII [MPa]")
            ax3 = Axis(fig[1,3], title = "log10(εII)")
            ax4 = Axis(fig[2,3], title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[3].*1e-3, Array(stokes.P[:, slice_j, :]./1e9) , colormap=:lajolla)
            # Plot particles phase
            h2  = heatmap!(ax2, xci[1].*1e-3, xci[3].*1e-3, Array(stokes.τ.II[:, slice_j, :].*1e-6) , colormap=:batlow)
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(stokes.ε.II[:, slice_j, :])) , colormap=:batlow)
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1].*1e-3, xci[3].*1e-3, Array(log10.(η[:, slice_j, :])) , colormap=:batlow)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[1,4], h3)
            Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
nx,ny,nz = 50, 50, 50
# nx,ny,nz = 128, 35, 101
li, origin, phases_GMG, = GMG_only(nx+1, ny+1, nz+1)
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI= true)...)
else
    igg
end

# (Path)/folder where output data and figures are stored
figdir   = "Subduction3D"
# main3D(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);