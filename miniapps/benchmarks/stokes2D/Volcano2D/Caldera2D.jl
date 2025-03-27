# const isCUDA = false
const isCUDA = true

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
using GeoParams, CairoMakie, CellArrays, Statistics, Dates, JLD2

# Load file with all the rheology configurations
include("Caldera_setup.jl")
include("Caldera_rheology.jl")

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
    @all(P) = abs(@all(ρg) * @all_k(z)) #* <(@all_k(z), 0.0)
    return nothing
end

function apply_pure_shear(Vx, Vy, εbg, xvi, lx, ly)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Vx, εbg, lx)
        xi = xv[i]
        Vx[i, j + 1] = εbg * (xi - lx * 0.5)
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Vy, εbg, ly)
        yi = yv[j]
        Vy[i + 1, j] = abs(yi) * εbg
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Vx, εbg, lx)
    nx, ny = size(Vy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Vy, εbg, ly)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

function extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    topo_idx = [findfirst(x -> x == air_phase, row) - 1 for row in eachrow(phases_GMG)]
    yv = xvi[2]
    topo_y = yv[topo_idx]
    return topo_y
end

function thermal_anomaly!(Temp, Ω_T, phase_ratios, T_chamber, T_air, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @parallel_indices (i, j) function _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, vertex_ratio, conduit_phase, magma_phase, anomaly_phase, air_phase)
        # quick escape
        conduit_ratio_ij = @index vertex_ratio[conduit_phase, i, j]
        magma_ratio_ij = @index vertex_ratio[magma_phase, i, j]
        anomaly_ratio_ij = @index vertex_ratio[anomaly_phase, i, j]
        air_ratio_ij = @index vertex_ratio[air_phase, i, j]

        # if conduit_ratio_ij > 0.5 || magma_ratio_ij > 0.5
        if conduit_ratio_ij > 0.5 || anomaly_ratio_ij > 0.5
            # if isone(conduit_ratio_ij) || isone(magma_ratio_ij)
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber
        elseif magma_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_chamber - 100.0e0
        elseif air_ratio_ij > 0.5
            Ω_T[i + 1, j] = Temp[i + 1, j] = T_air
        end

        return nothing
    end

    ni = size(phase_ratios.vertex)

    @parallel (@idx ni) _thermal_anomaly!(Temp, Ω_T, T_chamber, T_air, phase_ratios.vertex, conduit_phase, magma_phase, anomaly_phase, air_phase)

    @views Ω_T[1, :] .= Ω_T[2, :]
    @views Ω_T[end, :] .= Ω_T[end - 1, :]
    @views Temp[1, :] .= Temp[2, :]
    @views Temp[end, :] .= Temp[end - 1, :]

    return nothing
end

function plot_particles(particles, pPhases)
    p = particles.coords
    # pp = [argmax(p) for p in phase_ratios.center] #if you want to plot it in a heatmap rather than scatter
    ppx, ppy = p
    # pxv = ustrip.(dimensionalize(ppx.data[:], km, CharDim))
    # pyv = ustrip.(dimensionalize(ppy.data[:], km, CharDim))
    pxv = ppx.data[:]
    pyv = ppy.data[:]
    clr = pPhases.data[:]
    # clr = pϕ.data[:]
    idxv = particles.index.data[:]
    f, ax, h = scatter(Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = :roma, markersize = 1)
    Colorbar(f[1, 2], h)
    return f
end

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, T_GMG, igg; nx = 16, ny = 16, figdir = "figs2D", do_vtk = false, extension = 1.0e-15 * 0, cutoff_visc = (1.0e16, 1.0e23))

    # Physical domain ------------------------------------
    ni = nx, ny           # number of cells
    di = @. li / ni       # grid steps
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid             # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; incompressible = false, magma = false)
    rheology_incomp = init_rheologies(; incompressible = true, magma = false)
    dt_time = 1.0e3 * 3600 * 24 * 365
    κ = (4 / (rheology[1].HeatCapacity[1].Cp.val * rheology[1].Density[1].ρ0.val)) # thermal diffusivity                                 # thermal diffusivity
    dt_diff = 0.5 * min(di...)^2 / κ / 2.01
    dt = min(dt_time, dt_diff)
    # ----------------------------------------------------

    # randomize cohesion
    perturbation_C = @rand(ni...)
    # Initialize particles -------------------------------
    nxcell = 100
    max_xcell = 150
    min_xcell = 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    pPhases, pT = init_cell_arrays(particles, Val(2))

    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)

    # Initialize marker chain
    nxcell, max_xcell, min_xcell = 100, 150, 75
    initial_elevation = 0.0e0
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    air_phase = 6
    topo_y = extract_topo_from_GMG_phases(phases_GMG, xvi, air_phase)
    for _ in 1:3
        @views hn = 0.5 .* (topo_y[1:(end - 1)] .+ topo_y[2:end])
        @views topo_y[2:(end - 1)] .= 0.5 .* (hn[1:(end - 1)] .+ hn[2:end])
        fill_chain_from_vertices!(chain, PTArray(backend)(topo_y))
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    end
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # rock ratios for variational stokes
    # RockRatios
    ϕ = RockRatio(backend, ni)
    # update_rock_ratio!(ϕ, phase_ratios, air_phase)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, Re = 3.0e0, r = 0.7, CFL = 0.98 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    @views thermal.T[2:(end - 1), :] .= PTArray(backend)(T_GMG)

    # Add thermal anomaly BC's
    T_chamber = 1223.0e0
    T_air = 273.0e0
    Ω_T = @zeros(size(thermal.T)...)
    thermal_anomaly!(thermal.T, Ω_T, phase_ratios, T_chamber, T_air, 5, 3, 4, air_phase)
    # JustRelax.DirichletBoundaryCondition(Ω_T)

    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (; left = true, right = true, top = false, bot = false),
        # dirichlet    = (; mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    Ttop = thermal.T[2:(end - 1), end]
    Tbot = thermal.T[2:(end - 1), 1]
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(2))
    for _ in 1:5
        compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        @parallel init_P!(stokes.P, ρg[end], xvi[2])
    end
    # stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Melt fraction
    ϕ_m = @zeros(ni...)
    compute_melt_fraction!(
        ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
    )
    # Rheology
    args0 = (; ϕ = ϕ_m, T = thermal.Tc, P = stokes.P, dt = Inf, perturbation_C = perturbation_C)
    viscosity_cutoff = (1.0e17, 1.0e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, air_phase, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ = 1.0e-8, CFL = 0.95 / √2
    )

    # Boundary conditions
    # flow_bcs         = DisplacementBoundaryConditions(;
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        free_surface = false,
    )

    # U            = 0.02
    # stokes.U.Ux .= PTArray(backend)([(x - li[1] * 0.5) * U / dt for x in xvi[1], _ in 1:ny+2])
    # stokes.U.Uy .= PTArray(backend)([-y * U / dt for _ in 1:nx+2, y in xvi[2]])
    # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # displacement2velocity!(stokes, dt)

    εbg = extension
    apply_pure_shear(@velocity(stokes)..., εbg, xvi, li...)

    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if plotting
        if do_vtk
            vtk_dir = joinpath(figdir, "vtk")
            take(vtk_dir)
        end
        take(figdir)
        checkpoint = joinpath(figdir, "checkpoint")
    end
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

    ## Plot initial T and P profile
    fig = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(; size = (1200, 900))
        ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Density")
        scatter!(
            ax1,
            Array(thermal.T[2:(end - 1), :][:]),
            Yv ./ 1.0e3,
        )
        # lines!(
        scatter!(
            ax2,
            # Array(stokes.P[:]./1e6),
            Array(ρg[2][:] ./ 9.81),
            Y ./ 1.0e3,
        )
        hideydecorations!(ax2)
        # save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    τxx_v = @zeros(ni .+ 1...)
    τyy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    interval = 0
    iterMax = 150.0e3
    local iters
    thermal.Told .= thermal.T

    while it < 100 #000 # run only for 5 Myrs
        if it > 1 && iters.iter > iterMax && iters.err_evo1[end] > pt_stokes.ϵ * 5
            iterMax += 10.0e3
            iterMax = min(iterMax, 200.0e3)
            println("Increasing maximum pseudo timesteps to $iterMax")
        else
            iterMax = 150.0e3
        end

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= Ttop
        @views T_buffer[:, 1] .= Tbot
        # clamp!(T_buffer, 273e0, 1223e0)
        @views thermal.T[2:(end - 1), :] .= T_buffer
        if it > 1  && rem(it, 5) == 0
            # if mod(round(t/(1e3 * 3600 * 24 *365.25); digits=1), 1e3) == 0.0
            println("Simulation eruption at t = $(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 2)) Kyrs")
            thermal_anomaly!(thermal.T, Ω_T, phase_ratios, T_chamber, T_air, 5, 3, 4, air_phase)
            interval += 1
            copyinn_x!(T_buffer, thermal.T)
            @views T_buffer[:, end] .= Ttop
            @views T_buffer[:, 1] .= Tbot
            temperature2center!(thermal)
            grid2particle!(pT, xvi, T_buffer, particles)
        end
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        args = (; ϕ = ϕ_m, T = thermal.Tc, P = stokes.P, dt = Inf, ΔTc = thermal.ΔTc, perturbation_C = perturbation_C)
        # args = (; ϕ=ϕ_m, T=thermal.Tc, P=stokes.P, dt=Inf)

        stress2grid!(stokes, pτ, xvi, xci, particles)

        t_stokes = @elapsed solve_VariationalStokes!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            ϕ,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax = 100.0e3,
                nout = 2.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, xci, xvi, dt)

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("Extrema T[C]: $(extrema(thermal.T .- 273))")
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        dtmax = 2.0e3 * 3600 * 24 * 365.25
        dt = compute_dt(stokes, di, dtmax)

        println("dt = $(dt / (3600 * 24 * 365.25)) years")
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
                igg = igg,
                phase = phase_ratios,
                iterMax = 100.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        thermal.ΔT .= thermal.T .- thermal.Told
        vertex2center!(thermal.ΔTc, thermal.ΔT[2:(end - 1), :])

        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------

        # Advection --------------------
        copyinn_x!(T_buffer, thermal.T)
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        center2vertex!(τxx_v, stokes.τ.xx)
        center2vertex!(τyy_v, stokes.τ.yy)
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
            xvi
        )

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        compute_melt_fraction!(
            ϕ_m, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P)
        )

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        # update_rock_ratio!(ϕ, phase_ratios, air_phase)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        tensor_invariant!(stokes.τ)

        @show it += 1
        t += dt

        if it == 1
            stokes.EII_pl .= 0.0
        end

        if plotting
            # Data I/O and plotting ---------------------
            if it == 1 || rem(it, 1) == 0
                ## this is used for plotting for now:
                η_eff = @. stokes.τ.II / (2 * stokes.ε.II)
                ##
                if igg.me == 0 && it == 1
                    metadata(pwd(), checkpoint, basename(@__FILE__), joinpath(@__DIR__, "Caldera_setup.jl"), joinpath(@__DIR__, "Caldera_rheology.jl"))
                end
                checkpointing_jld2(checkpoint, stokes, thermal, t, dt, igg)
                checkpointing_particles(checkpoint, particles; phases = pPhases, phase_ratios = phase_ratios, chain = chain, particle_args = particle_args, t = t, dt = dt)
                (; η_vep, η) = stokes.viscosity
                if do_vtk
                    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                    data_v = (;
                        T = Array(T_buffer),
                        stress_xy = Array(stokes.τ.xy),
                        strain_rate_xy = Array(stokes.ε.xy),
                        phase_vertices = [argmax(p) for p in Array(phase_ratios.vertex)],
                    )
                    data_c = (;
                        P = Array(stokes.P),
                        viscosity = Array(η_eff),
                        phases = [argmax(p) for p in Array(phase_ratios.center)],
                        Melt_fraction = Array(ϕ_m),
                        EII_pl = Array(stokes.EII_pl),
                        stress_II = Array(stokes.τ.II),
                        strain_rate_II = Array(stokes.ε.II),
                        plastic_strain_rate_II = Array(stokes.ε_pl.II),
                        density = Array(ρg[2] ./ 9.81),
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
                        velocity_v;
                        t = round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)
                    )
                    save_marker_chain(joinpath(vtk_dir, "chain_" * lpad("$it", 6, "0")), xvi[1], Array(chain.h_vertices))
                end

                # Make particles plottable
                p = particles.coords
                ppx, ppy = p
                pxv = ppx.data[:] ./ 1.0e3
                pyv = ppy.data[:] ./ 1.0e3
                clr = pPhases.data[:]
                # clr      = pT.data[:]
                idxv = particles.index.data[:]

                chain_x = chain.coords[1].data[:] ./ 1.0e3
                chain_y = chain.coords[2].data[:] ./ 1.0e3

                # Make Makie figure
                ar = DataAspect()
                fig = Figure(size = (1200, 900), title = "t = $t")
                ax1 = Axis(fig[1, 1], aspect = ar, title = "T [C]  (t=$(round(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs)")
                ax2 = Axis(fig[2, 1], aspect = ar, title = "Vy [cm/yr]")
                # ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
                ax3 = Axis(fig[1, 3], aspect = ar, title = "τII [MPa]")
                # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(εII)")
                ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
                ax5 = Axis(fig[3, 1], aspect = ar, title = "EII_pl")
                ax6 = Axis(fig[3, 3], aspect = ar, title = "Melt fraction ϕ")
                # Plot temperature
                h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), :] .- 273), colormap = :batlow)
                # Plot particles phase

                h2 = heatmap!(ax2, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, uconvert.(u"cm/yr", Array(stokes.V.Vy)u"m/s"), colormap = :batlow)
                scatter!(ax2, Array(chain_x), Array(chain_y), color = :red, markersize = 3)

                # Plot 2nd invariant of strain rate
                # h3  = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
                h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.τ.II) ./ 1.0e6, colormap = :batlow)
                # Plot effective viscosity
                # h4  = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:lipari)
                h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(η_eff)), colorrange = log10.(viscosity_cutoff), colormap = :batlow)
                h5 = heatmap!(ax5, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(stokes.EII_pl), colormap = :batlow)
                h6 = heatmap!(ax6, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(ϕ_m), colormap = :lipari)
                hidexdecorations!(ax1)
                hidexdecorations!(ax2)
                hidexdecorations!(ax3)
                hidexdecorations!(ax4)
                hideydecorations!(ax3)
                hideydecorations!(ax4)
                hideydecorations!(ax6)

                Colorbar(fig[1, 2], h1)
                Colorbar(fig[2, 2], h2)
                Colorbar(fig[1, 4], h3)
                Colorbar(fig[2, 4], h4)
                Colorbar(fig[3, 2], h5)
                Colorbar(fig[3, 4], h6)
                linkaxes!(ax1, ax2, ax3, ax4, ax5)
                fig
                save(joinpath(figdir, "$(it).png"), fig)

                # ## Plot initial T and P profile
                fig = let
                    Yv = [y for x in xvi[1], y in xvi[2]][:]
                    Y = [y for x in xci[1], y in xci[2]][:]
                    fig = Figure(; size = (1200, 900))
                    ax1 = Axis(fig[1, 1]; aspect = 2 / 3, title = "T")
                    ax2 = Axis(fig[1, 2]; aspect = 2 / 3, title = "Pressure")
                    scatter!(
                        ax1,
                        Array(thermal.T[2:(end - 1), :][:] .- 273.15),
                        Yv ./ 1.0e3,
                    )
                    lines!(
                        ax2,
                        Array(stokes.P[:] ./ 1.0e6),
                        Y ./ 1.0e3,
                    )
                    hideydecorations!(ax2)
                    # save(joinpath(figdir, "thermal_profile_$it.png"), fig)
                    fig
                end

                # Plot Drucker Prager yield surface
                fig1 = let
                    fig = Figure(; size = (1200, 900))
                    ax = Axis(fig[1, 1]; title = "Drucker Prager")
                    lines!(ax, [0.0e6, maximum(stokes.P)] ./ 1.0e6, [10.0e6; (maximum(stokes.P) * sind(rheology[1].CompositeRheology[1].elements[end].ϕ.val) - rheology[1].CompositeRheology[1].elements[end].C.val * cosd(rheology[1].CompositeRheology[1].elements[end].ϕ.val))] ./ 1.0e6, color = :black, linewidth = 2)
                    s1 = scatter!(ax, Array(stokes.P ./ 1.0e6)[:], Array(stokes.τ.II ./ 1.0e6)[:]; color = Array(stokes.R.RP)[:], colormap = :roma, markersize = 3)
                    Colorbar(fig[1, 2], s1)
                    fig
                    save(joinpath(figdir, "DruckerPrager_$it.png"), fig)
                end
            end
            # ------------------------------
        end
    end

    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------
const plotting = true

do_vtk = true # set to true to generate VTK files for ParaView

figdir = "Caldera2D_$(today())"
n = 256
nx, ny = n, n >>> 1

li, origin, phases_GMG, T_GMG = setup2D(
    nx + 1, ny + 1;
    sticky_air = 4.0e0,
    dimensions = (25.0e0, 20.0e0), # extent in x and y in km
    flat = false, # flat or volcano cone
    chimney = true, # conduit or not
    volcano_size = (3.0e0, 6.0e0),    # height, radius
    conduit_radius = 4.0e-1, # radius of the conduit
    chamber_T = 900.0e0, # temperature of the chamber
    chamber_depth = 5.0e0, # depth of the chamber
    chamber_radius = 1.25e0, # radius of the chamber
    aspect_x = 2.5e0, # aspect ratio of the chamber
)

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
extension = 1.0e-15
main(li, origin, phases_GMG, T_GMG, igg, ; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk, extension = extension, cutoff_visc = (1.0e17, 1.0e23));
