using CUDA
CUDA.allowscalar(false)
using JustRelax, JustRelax.DataIO
import JustRelax.@cell
using ParallelStencil
@init_parallel_stencil(CUDA, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:CUDA, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CairoMakie, CellArrays, WriteVTK, Statistics

# Load file with all the rheology configurations
include("Sill_rheology_small_sill.jl")


## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function copyinn_x!(A, B)

    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, x)
    depth = y[j]

    T[i + 1 , j] = 273e0 + 600e0

    if (-0.09e3 < depth ≤  -0.03e3)
        T[i + 1, j] = 273e0 + 1200e0
    end

    if  (-0.06e3 < depth ≤  -0.05e3 ) && (120 < x[i] ≤  130)
        T[i + 1, j] = 273e0 + 1300e0
    end


    return nothing
end

@parallel_indices (i, j) function compute_melt_fraction!(ϕ, rheology, args)
    ϕ[i, j] = compute_meltfraction(rheology, ntuple_idx(args, i, j))
    return nothing
end

@parallel_indices (I...) function compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
args_ijk = ntuple_idx(args, I...)
ϕ[I...] = compute_melt_frac(rheology, args_ijk, phase_ratios[I...])
return nothing
end

@inline function compute_melt_frac(rheology, args, phase_ratios)
    return GeoParams.compute_meltfraction_ratio(phase_ratios, rheology, args)
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar=8, ny=16, nx=ny*8, figdir="figs2D", do_save_vtk =false)

    # Physical domain ------------------------------------
    ly           = 0.125e3            # domain length in y
    lx           = 0.25e3             # domain length in x
    ni           = nx, ny            # number of cells
    li           = lx, ly            # domain length in x- and y-
    di           = @. li / ni        # grid step in x- and -y
    origin       = 0,-ly             # origin coordinates (15km f sticky air layer)
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology     = init_rheologies(; is_plastic = false)
    κ            = (4 / (compute_heatcapacity(rheology[1].HeatCapacity[1].Cp) * rheology[1].Density[1].ρ))
    @show κ
    # κ            = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 / 100 # diffusive CFL timestep limiter
   # dt = dt_diff/10
    @show dt
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 30, 45, 15
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )

    subgrid_arrays = SubgridDiffusionCellArrays(particles)

    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases      = init_cell_arrays(particles, Val(3))
    particle_args    = (pT, pPhases)

    # initialize phases
    init_phases!(pPhases, particles)
    phase_ratios     = PhaseRatio(ni, length(rheology))
    # @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(ni, ViscoElastic)
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=5e-5,  CFL = 0.95 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal          = ThermalArrays(ni)
    thermal_bc       = TemperatureBoundaryConditions(;
        no_flux      = (left = true, right = true, top = false, bot = false),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2], xvi[1])
    thermal_bcs!(thermal.T, thermal_bc)

    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)
    # ----------------------------------------------------
    #melt fraction
    ϕ                = @zeros(ni...)
    @parallel (@idx ni) compute_melt_fraction!(
        ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
    )

    # Buoyancy forces
    ρg               = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P, ϕ= ϕ))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end
    # Rheology
    η                = @ones(ni...)
    args             = (; T = thermal.Tc, P = stokes.P, dt = Inf, ϕ=ϕ)
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )
    η_vep            = copy(η)
    # ϕ                = similar(η)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL= 5e-2 / √2.1
    )

    # Boundary conditions
    flow_bcs         = FlowBoundaryConditions(;
        free_slip    = (left = true, right=true, top=true, bot=true),
        periodicity  = (left = false, right = false, top = false, bot = false),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_save_vtk
        vtk_dir      = joinpath(figdir,"vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv  = [y for x in xvi[1], y in xvi[2]][:]
        Y   = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        scatter!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, minimum(xvi[2])./1e3, 0)
        ylims!(ax2, minimum(xvi[2])./1e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer    = @zeros(ni.+1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    local Vx_v, Vy_v
    if do_save_vtk
        Vx_v = @zeros(ni.+1...)
        Vy_v = @zeros(ni.+1...)
    end
    # Time loop
    t, it = 0.0, 0


    while it < 30e3
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P,  dt=dt, ϕ= ϕ)
        @parallel (@idx ni) compute_viscosity!(
            η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
        )
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        solve!(
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
            iterMax = 150e3,
            nout=1e3,
            viscosity_cutoff=(-Inf, Inf),
            verbose = true,
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        dt   = compute_dt(stokes, di, dt_diff)

        # ------------------------------

    #     # interpolate fields from particle to grid vertices
    #     particle2grid!(T_buffer, pT, xvi, particles)
    #    # @views T_buffer[:, end]      .= 273.0
    #     @views T_buffer[:, end]      .= minimum(T_buffer)
    #     @views thermal.T[2:end-1, :] .= T_buffer
    #     temperature2center!(thermal)

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
            iterMax = 100e3,
            nout    = 1e2,
            verbose = true,
        )
        # ------------------------------

        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:end-1, :], subgrid_arrays, particles, xvi,  di, dt
        )

        # Advection --------------------
        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # interpolate fields from grid vertices to particles
        # grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
        # check if we need to inject particles
        inject = check_injection(particles)
        inject && inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer,), xvi)

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views thermal.T[:, end]     .= 273e0 + 600e0
        @views thermal.T[:, 1]       .= 273e0 + 600e0
        @views thermal.T[2:end-1, :] .= T_buffer
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        thermal_bcs!(thermal.T, thermal_bc)
        temperature2center!(thermal)

        @show extrema(thermal.T)
        any(isnan.(thermal.T)) && break


        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, particles.coords, xci, di, pPhases)
        # @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)
        @parallel (@idx ni) compute_melt_fraction!(
            ϕ, phase_ratios.center, rheology, (T=thermal.Tc, P=stokes.P)
        )
        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 25) == 0
            checkpointing(figdir, stokes, thermal.T, η, t)

            if do_save_vtk
                JustRelax.velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

                data_v = (;
                    T   = Array(thermal.T[2:end-1, :]),
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
                    ϕ   = Array(ϕ),
                    ρ  = Array(ρg[2]./10),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                )
            end

            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];

            # Make Makie figure
            fig = Figure(size = (2000, 1000), title = "t = $t", )
            ax0 = Axis(
                fig[1, 1:2];
                aspect=ar,
                title="t=$(round((t/(3600)))) hours",
                titlesize=50,
                height=0.0,
            )
            ax0.ylabelvisible = false
            ax0.xlabelvisible = false
            ax0.xgridvisible = false
            ax0.ygridvisible = false
            ax0.xticksvisible = false
            ax0.yticksvisible = false
            ax0.yminorticksvisible = false
            ax0.xminorticksvisible = false
            ax0.xgridcolor = :white
            ax0.ygridcolor = :white
            ax0.ytickcolor = :white
            ax0.xtickcolor = :white
            ax0.yticklabelcolor = :white
            ax0.xticklabelcolor = :white
            ax0.yticklabelsize = 0
            ax0.xticklabelsize = 0
            ax0.xlabelcolor = :white
            ax0.ylabelcolor = :white

            ax1 = Axis( fig[2, 1][1, 1], aspect = DataAspect(), title = L"T \;[\mathrm{C}]",  titlesize=40,
            yticklabelsize=25,
            xticklabelsize=25,
            xlabelsize=25,)
            #ax2 = Axis(fig[2,1], aspect = DataAspect(), title = "Phase")
            ax2 = Axis(fig[2, 2][1, 1], aspect = DataAspect(), title = L"Density \;[\mathrm{kg/m}^{3}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)


            #ax3 = Axis(fig[1,3], aspect = DataAspect(), title = "log10(εII)")
            ax3 = Axis(fig[3, 1][1, 1], aspect = DataAspect(), title = L"Vy \;[\mathrm{m/s}]", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            #ax4 = Axis(fig[2,3], aspect = DataAspect(), title = "log10(η)")
            ax4 = Axis(fig[3, 2][1, 1], aspect = DataAspect(), title = L"\phi", titlesize=40,
                    yticklabelsize=25,
                    xticklabelsize=25,
                    xlabelsize=25,)

            # Plot temperature
            h1  = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T[2:end-1,:].- 273.15) , colormap=:lipari, colorrange=(700, 1200))
            # Plot particles phase
            #h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))
            #h2  = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]))

            h2  = heatmap!(ax2, xci[1], xci[2], Array(ρg[2]./10.0) , colormap=:batlowW, colorrange=(2650, 2820))

            # Plot 2nd invariant of strain rate
            #h3  = heatmap!(ax3, xci[1], xci[2], Array(log10.(stokes.ε.II)) , colormap=:batlow)

            # Plot vy velocity
            h3  = heatmap!(ax3, xvi[1], xvi[2], Array(stokes.V.Vy) , colormap=:batlow)

            # Plot effective viscosity
            #h4  = heatmap!(ax4, xci[1], xci[2], Array(log10.(η_vep)) , colormap=:batlow)

            # Plot melt fraction
            h4  = heatmap!(ax4, xci[1], xci[2], Array(ϕ) , colormap=:lipari)


            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hideydecorations!(ax2)
            hideydecorations!(ax4)
            Colorbar(fig[2, 1][1, 2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[2, 2][1, 2], h2, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 1][1, 2], h3, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            Colorbar(fig[3, 2][1, 2], h4, height = Relative(4/4), ticklabelsize=25, ticksize=15)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig

            let
                fig = Figure(size = (2000, 1000), title = "t = $t")
                ax1 = Axis(fig[1,1], aspect = DataAspect(), title = "T [C]  (t=$(round((t/(3600)))) hours)",  titlesize=40,
                yticklabelsize=25,
                xticklabelsize=25,
                xlabelsize=25,)
                 # Plot temperature
                h1  = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T[2:end-1,:].- 273.15) , colormap=:lipari, colorrange=(700, 1200))
                Colorbar(fig[1,2], h1, height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Temperature_$(it).png"), fig)
                fig
            end
            let
                fig = Figure(size = (2000, 1000), title = "t = $t")
                ax1 = Axis(fig[1,1], aspect = DataAspect(), title = "Particles  (t=$(round((t/(3600)))) hours)",  titlesize=40,
                yticklabelsize=25,
                xticklabelsize=25,
                xlabelsize=25,)
                p = particles.coords
                ppx, ppy = p
                pxv = ppx.data[:]
                pyv = ppy.data[:]
                clr = pPhases.data[:]
                idxv = particles.index.data[:]
                h=scatter!(ax1,Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), colormap=:roma, markersize=3)
                Colorbar(fig[1,2], h,height = Relative(4/4), ticklabelsize=25, ticksize=15)
                save(joinpath(figdir, "Particles_$(it).png"), fig)
                fig
            end

        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------


# (Path)/folder where output data and figures are stored
figdir   = "Overnight_240321_more_particles_eta_1e5_bas_1e3_rhy"
# figdir   = "test_JP"
do_save_vtk = true # set to true to generate VTK files for ParaView
ar       = 1 # aspect ratio
n        = 320
nx       = n*ar - 2
ny       = n - 2
igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, do_save_vtk = do_save_vtk);
