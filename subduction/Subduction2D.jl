using CUDA
using JustRelax, JustRelax.DataIO
import JustRelax.@cell
using ParallelStencil
@init_parallel_stencil(CUDA, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
# const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
# model = PS_Setup(:Threads, Float64, 2) # or (:CUDA, Float64, 3) or (:AMDGPU, Float64, 3)
model = PS_Setup(:CUDA, Float64, 2) # or (:CUDA, Float64, 3) or (:AMDGPU, Float64, 3)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, CellArrays

# Load file with all the rheology configurations
include("Subduction_rheology2D.jl")
include("GMG_setup2D.jl")

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
    rheology     = init_rheologies()
    dt           = 10e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell          = 40
    max_xcell       = 60
    min_xcell       = 20
    particles       = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays  = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi        = velocity_grids(xci, xvi, di)
    # temperature
    pPhases, pT     = init_cell_arrays(particles, Val(2))
    particle_args   = (pPhases, pT)

    # Assign particles phases anomaly
    phases_device    = PTArray(phases_GMG)
    init_phases!(pPhases, phases_device, particles, xvi)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    phase_ratios_center!(phase_ratios, particles, xci, di, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    @views thermal.T[2:end-1, :] .= PTArray(T_GMG)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    @parallel (@idx ni) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------
    
    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    for _ in 1:2
        compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
        JustRelax.Stokes2D.init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    η                = @ones(ni...)
    η_vep            = similar(η)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(
        η, 1.0, phase_ratios, stokes, args, rheology, (1e18, 1e24)
    )

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=1e-2 / √3
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = true,
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # Plot initial T and η profiles
    let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        # scatter!(ax2, Array(ρg[2][:]), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

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
    while it < 1000 # run only for 5 Myrs
    # while (t/(1e6 * 3600 * 24 *365.25)) < 5 # run only for 5 Myrs
        
        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end]      .= 273.0
        @views T_buffer[:, 1]        .= 1623.0
        @views thermal.T[2:end-1, :] .= T_buffer
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        
        # interpolate fields from particle to grid vertices
        # particle2grid!(thermal.T, pT, xvi, particles)
        # temperature2center!(thermal)
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)
        compute_viscosity!(
            η, 1.0, phase_ratios, stokes, args, rheology, (1e18, 1e24)
        )
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        
        # Stokes solver ----------------
        t_stokes = @elapsed begin
            out = solve!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                η,
                η_vep,
                phase_ratios,
                rheology,
                args,
                dt,
                igg;
                iterMax          = 150e3,
                nout             = 2e3,
                viscosity_cutoff = (1e18, 1e24),
                free_surface = true,
                # viscosity_relaxation = 1e-5
            );
        end
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt   = compute_dt(stokes, di)
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
            igg     = igg,
            phase   = phase_ratios,
            iterMax = 10e3,
            nout    = 1e2,
            verbose = true,
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
        advection_RK!(particles, @velocity(stokes), grid_vxi..., dt, 2 / 3)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if do_vtk
                JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(thermal.T),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    P   = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η   = Array(η),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
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
            h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
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
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "Subduction2D"
# nx, ny   = 512, 256
nx, ny   = 512, 128
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx+1, ny+1)
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
