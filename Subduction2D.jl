const isCUDA = false

@static if isCUDA 
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.@cell
using ParallelStencil, ParallelStencil.FiniteDifferences2D
@static if isCUDA 
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

const backend = @static if isCUDA 
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA 
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("Subduction2D_rheology.jl")
include("Subduction2D_GMG.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    # Physical domain ------------------------------------
    ni           = nx, ny           # number of cells
    di           = @. li / ni       # grid steps
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    ρbg                = 2.7e3
    rheology_dev       = init_rheologies(; ρbg = ρbg)
    rheology           = init_rheologies(; ρbg = 0e0)
    dt                 = 50e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell          = 40
    max_xcell       = 60
    min_xcell       = 20
    particles       = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays  = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi        = velocity_grids(xci, xvi, di)
    # temperature
    pPhases, pT         = init_cell_arrays(particles, Val(2))
    # τxx_p, τyy_p, τxy_p = init_cell_arrays(particles, Val(3))
    # vorticity_p,        = init_cell_arrays(particles, Val(1))
    # particle_args       = (pT, τxx_p, τyy_p, τxy_p, vorticity_p, pPhases)
    particle_args       = (pT, pPhases)
    
    # Assign particles phases anomaly
    phases_device    = PTArray(backend)(phases_GMG)
    init_phases!(pPhases, phases_device, particles, xvi)
    phase_ratios     = PhaseRatio(backend, ni, length(rheology))
    phase_ratios_center(phase_ratios, particles, grid, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=1e0, CFL = 1 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(backend, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)
    Ttop, Tbot       = extrema(T_GMG)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop 
    @views thermal.T[:, 1]   .= Tbot 
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    args  = (; T = thermal.Tc, P = zeros(ni...), Plitho = zeros(ni...), dt = Inf)
    ρg    = ntuple(_ -> @zeros(ni...), Val(2))
    # for _ in 1:2
    #     compute_ρg!(ρg[2], phase_ratios, rheology, args)
    #     # JustRelax.Stokes2D.init_P!(stokes.P, ρg[2], xci[2])
    # end
    compute_ρg!(ρg[2], phase_ratios, rheology_dev, args)
    ρg_bg   = ρbg * 9.81
    args.P .= reverse(cumsum(reverse((ρg[2] .+ ρg_bg).* di[2], dims=2), dims=2), dims=2)

    # Rheology
    viscosity_cutoff = (1e18, 1e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
   
    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-3 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = false,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Plot initial T and η profiles
    # let
    #     Yv  = [y for x in xvi[1], y in xvi[2]][:]
    #     Y   = [y for x in xci[1], y in xci[2]][:]
    #     fig = Figure(size = (1200, 900))
    #     ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
    #     ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
    #     scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
    #     # scatter!(ax2, Array(log10.(η[:])), Y./1e3)
    #     scatter!(ax2, Array(stokes.P[:]), Y./1e3)
    #     # scatter!(ax2, Array(Plitho[:]), Y./1e3)
    #     ylims!(ax1, minimum(xvi[2])./1e3, 0)
    #     ylims!(ax2, minimum(xvi[2])./1e3, 0)
    #     hideydecorations!(ax2)
    #     # save(joinpath(figdir, "initial_profile.png"), fig)
    #     fig
    # end

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir      = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀         = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    # Time loop
    t, it = 0.0, 0

    # Vertice arrays of normal components of the stress tensor
    # τxx_vertex, τyy_vertex = @zeros(ni.+1...), @zeros(ni.+1...)
    # τxx_o_vertex, τyy_o_vertex = @zeros(ni.+1...), @zeros(ni.+1...)

    while it < 1000 # run only for 5 Myrs
    # while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs
        
        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= Ttop
        @views T_buffer[:, 1]        .= Tbot
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        
        # Update buoyancy and viscosity -
        args.Plitho .= reverse(cumsum(reverse((ρg[2] .+ ρg_bg).* di[2], dims=2), dims=2), dims=2)
        args.P      .= stokes.P .- minimum(stokes.P) .+ args.Plitho 
    
        # args = (; T = thermal.Tc, P = Plitho,  dt=Inf)
        # args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
        compute_ρg!(ρg[2], phase_ratios, rheology_dev, args)
        
        # Stokes solver ----------------
        t_stokes = @elapsed begin
            out = solve!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                rheology_dev,
                args,
                dt,
                igg;
                kwargs = (
                    iterMax              = 50e3,
                    nout                 = 1e3,
                    viscosity_cutoff     = (1e18, 1e24),
                    free_surface         = false,
                    viscosity_relaxation = 1e-2
                )
            );
        end
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di)
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (
                igg     = igg,
                phase   = phase_ratios,
                iterMax = 150e3,
                nout    = 2e3,
                verbose = true
            ),
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # JustRelax.Stokes2D.rotate_stress_particles!(
        #     stokes,
        #     τxx_vertex, 
        #     τyy_vertex,
        #     τxx_o_vertex, 
        #     τyy_o_vertex,
        #     τxx_p, 
        #     τyy_p, 
        #     τxy_p,
        #     vorticity_p,
        #     particles,
        #     grid,
        #     dt
        # )

        # check if we need to inject particles
        # inject_particles_phase!(particles, pPhases, (pT, τxx_p, τyy_p, τxy_p, vorticity_p), (T_buffer, τxx_vertex, τyy_vertex, stokes.τ.xy, stokes.ω.xy_v), xvi)
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        phase_ratios_center(phase_ratios, particles, grid, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η, η_vep) = stokes.viscosity
            # if do_vtk
            #     velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            #     data_v = (;
            #         T   = Array(T_buffer),
            #         τxy = Array(stokes.τ.xy),
            #         εxy = Array(stokes.ε.xy),
            #         Vx  = Array(Vx_v),
            #         Vy  = Array(Vy_v),
            #     )
            #     data_c = (;
            #         P   = Array(stokes.P),
            #         τxx = Array(stokes.τ.xx),
            #         τyy = Array(stokes.τ.yy),
            #         εxx = Array(stokes.ε.xx),
            #         εyy = Array(stokes.ε.yy),
            #         η   = Array(η_vep),
            #     )
            #     velocity_v = (
            #         Array(Vx_v),
            #         Array(Vy_v),
            #     )
            #     save_vtk(
            #         joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
            #         xvi,
            #         xci,
            #         data_v,
            #         data_c,
            #         velocity_v
            #     )
            # end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            # clr      = pT.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            ar  = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1  = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # Plot particles phase
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            # Plot effective viscosity
            h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[1,4], h3)
            Colorbar(fig[2,4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------
    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "Subduction_LAMEM_2D"
# nx, ny   = 512, 256
# nx, ny   = 512, 128
n        = 64
nx, ny   = n*6, n
# nx, ny   = (512, 128)
nx, ny   = 256, 64
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx+1, ny+1)
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
