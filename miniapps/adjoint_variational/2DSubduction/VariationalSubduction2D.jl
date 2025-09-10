const isCUDA = false
#const isCUDA = true

using Statistics

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CellArrays
using CairoMakie

# Load file with all the rheology configurations
include("Subduction2D_setup.jl")
include("VariationalSubduction2D_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    return esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

function custom_argmax(p)
    if p[1] != 0.0
        return 0
    else
        return argmax(p)
    end
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
 function main(li, origin, phases_GMG, igg; nx::Int64 = 16, ny::Int64 = 16, figdir::String = "figs2D", do_vtk::Bool = false)

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = 10.0e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 100
    max_xcell = 125
    min_xcell = 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)

    # particle fields for the stress rotation
    pPhases, pT = init_cell_arrays(particles, Val(2))
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # marker chain
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)

    # RockRatios
    air_phase = 7
    ϕ_R = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ_R, chain, xvi, di)
    #update_rock_ratio!(ϕ_R, phase_ratios, air_phase)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    # pt_stokes = PTStokesCoeffs(li, di; ϵ_rel = 1.0e-6, Re = 15π, r = 0.7, CFL = 0.98 / √2.1)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-1, ϵ_rel = 1.0e-3, Re = 40π, r = 0.7, CFL = 0.98 / √2.1)
    # ----------------------------------------------------


    # TEMPERATURE PROFILE --------------------------------
    Ttop = 20 + 273
    Tbot = maximum(T_GMG)
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1] .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    stokes.P .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]) .* di[2], dims = 2), dims = 2), dims = 2))

    # Rheology
    args = (T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf); air_phase = air_phase)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
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

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # PT coefficients for thermal diffusion
    args0 = (T = thermal.Tc, P = stokes.P, dt = Inf)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # initilaize vtk collection]
    Vx_c  = @zeros(ni...)
    Vy_c  = @zeros(ni...)
    V_mag = @zeros(ni...)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    vtkc = VTKDataSeries(joinpath(figdir, "vtk_series"), xci)
    vtkv = VTKDataSeries(joinpath(figdir, "vtk_series"), xvi)
    

    # Time loop
    t, it = 0.0, 0
    dt = 25.0e3 * (3600 * 24 * 365.25)
    dt_max = 250.0e3 * (3600 * 24 * 365.25) /10.0

    AD_out = 10 # output every AD_out iterations
    while it < 500 # run only for 5 Myrs

        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)

            #    # interpolate fields from particle to grid vertices
            #    particle2grid!(T_buffer, pT, xvi, particles)
            #    @views T_buffer[:, end] .= Ttop
            #    @views T_buffer[:, 1] .= Tbot
            #    @views thermal.T[2:(end - 1), :] .= T_buffer
            #    thermal_bcs!(thermal, thermal_bc)
            #    temperature2center!(thermal)

        print("#################\n")
        print(extrema(thermal.T), "\n")
        print(mean(thermal.T), "\n")
        print("#################\n")

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ_R,
                rheology,
                args,
                dt,
                igg;
                kwargs = (;
                    iterMax = 400.0e3,
                    free_surface = true,
                    nout = 5.0e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end
        print("#################\n")
        print(extrema(thermal.T), "\n")
        print(mean(thermal.T), "\n")
        print("#################\n")

    	println("   Time/iteration:  $(t_stokes / it) s")
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_max)*0.4
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(dt / (3600 * 24 * 365.25)) kyrs")


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
                igg = igg,
                phase = phase_ratios,
                iterMax = 50.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------
    #    print("#################\n")
    #    print(extrema(thermal.T), "\n")
    #    print(mean(thermal.T), "\n")
    #    print("#################\n")
        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), (), xvi)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        compute_rock_fraction!(ϕ_R, chain, xvi, di)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, AD_out) == 0
            (; η_vep, η) = stokes.viscosity
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
        
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            Vx_c .= 0.0
            Vy_c .= 0.0
            vertex2center!(Vx_c, Vx_v)
            vertex2center!(Vy_c, Vy_v)
            V_mag .= sqrt.((Vx_c.^2 .+ Vy_c.^2))
        
            # Make particles plottable
            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            idxv     = particles.index.data[:];
            #CUDA.allowscalar() do
            pp = [custom_argmax(p) for p in phase_ratios.center]
            #end
        
            DataIO.append!(vtkc, (Phase=pp,T = thermal.T[2:end-1,:],P=Array(stokes.P), Vx=Array(Vx_c), Vy=Array(Vy_c), ρ=Array((ρg[2]./9.81)), η=Array(η), η_vep=Array(η_vep),  τII=Array(stokes.τ.II), εII=Array(stokes.ε.II), EII_pl=Array(stokes.EII_pl)), it, (t/(1e6 * 3600 * 24 *365.25)))

            #η_sens=Array(stokesAD.η), ρ_sens=Array(stokesAD.ρ),G_sens=Array(stokesAD.G), K_sens=Array(stokesAD.K)),
        end

        #=
        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                )
                data_c = (;
                    P = Array(stokes.P),
                    η = Array(η_vep),
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
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            # clr      = pT.data[:]
            idxv = particles.index.data[:]

            chain_x = Array(chain.coords[1].data)[:] ./ 1.0e3
            chain_y = Array(chain.coords[2].data)[:] ./ 1.0e3

            # Make Makie figure
            ar = 3
            fig = Figure(size = (1200, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "log10(εII)  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Phase")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "τII")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            #stokes.ε.II .= rand(1)
            h1 = heatmap!(ax1, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.ε.II)), colormap = :batlow)
            scatter!(ax1, chain_x, chain_y, markersize = 3)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 1)
            # Plot 2nd invariant of strain rate
            #stokes.τ.II .= rand(1)
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array((stokes.τ.II)), colormap = :batlow)
            scatter!(ax3, chain_x, chain_y, markersize = 3, color = :red)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep)), colormap = :batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            # display(fig)
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------
        =#

    end

    return
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk = true # set to true to generate VTK files for ParaView
figdir = "Subduction2D_MQS_variational"
nx, ny = 250, 100
# nx, ny = 25, 10
li, origin, phases_GMG, T_GMG = GMG_subduction_2D(nx + 1, ny + 1)
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
