# Benchmark of Duretz et al. 2014
# http://dx.doi.org/10.1002/2014GL060438
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using GeoParams, GLMakie

# Load file with all the rheology configurations
include("Shearheating_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    return @parallel f_x(A, B)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end


## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ly = 40.0e3       # domain length in y
    lx = 70.0e3       # domain length in x
    ni = nx, ny     # number of cells
    li = lx, ly     # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    origin = 0.0, -ly   # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid       # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; is_TP_Conductivity = false)
    κ = (4 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 32, 12
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(3))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2 # origin of thermal anomaly
    yc_anomaly = 40.0e3   # origin of thermal anomaly
    r_anomaly = 3.0e3    # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, xc_anomaly, yc_anomaly, r_anomaly)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )

    # Initialize constant temperature
    @views thermal.T .= 273.0 + 400
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])

    # Rheology
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 1.0e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    ## Compression and not extension - fix this
    εbg = 5.0e-14
    stokes.V.Vx .= PTArray(backend_JR)([ -(x - lx / 2) * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([ (ly - abs(y)) * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)


    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:(end - 1), :][:]), Yv ./ 1.0e3)
        scatter!(ax2, Array(log10.(stokes.viscosity.η[:])), Y ./ 1.0e3)
        ylims!(ax1, minimum(xvi[2]) ./ 1.0e3, 0)
        ylims!(ax2, minimum(xvi[2]) ./ 1.0e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    # Time loop
    t, it = 0.0, 0
    while it < 1
        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, (-Inf, Inf)
        )
        # ------------------------------

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (;
                iterMax = 75.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= 273.0 + 400
        @views thermal.T[2:(end - 1), :] .= T_buffer
        temperature2center!(thermal)

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # interpolate fields from grid vertices to particles
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        grid2particle_flip!(pT, xvi, T_buffer, Told_buffer, particles)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 10) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(thermal.T[2:(end - 1), :]),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                )
                data_c = (;
                    P = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η = Array(stokes.viscosity.η_vep),
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
                    velocity_v,
                    t = t
                )
            end

            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:] ./ 1.0e3
            pyv = ppy.data[:] ./ 1.0e3
            clr = pPhases.data[:]
            idxv = particles.index.data[:]

            # Make Makie figure
            fig = Figure(size = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [C]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Shear heating [W/m3]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.T[2:(end - 1), :] .- 273.0), colormap = :batlow)
            # Plot particles phase
            h2 = heatmap!(ax2, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(thermal.shear_heating), colormap = :batlow)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, Array(log10.(stokes.ε.II)), colormap = :batlow)
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
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return nothing
end

figdir = "Benchmark_Duretz_etal_2014"
do_vtk = false # set to true to generate VTK files for ParaView
ar = 1 # aspect ratio
n = 64
nx = n * ar - 2
ny = n - 2
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main2D(igg; ar = ar, ny = ny, nx = nx, figdir = figdir, do_vtk = do_vtk)
