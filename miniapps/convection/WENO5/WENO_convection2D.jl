using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions, CellArrays

# Load file with all the rheology configurations
include("Layered_rheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

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

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, z)
    depth = -z[j]

    # (depth - 15e3) because we have 15km of sticky air
    if depth < 0.0e0
        T[i + 1, j] = 273.0e0

    elseif 0.0e0 ≤ (depth) < 35.0e3
        dTdZ = (923 - 273) / 35.0e3
        offset = 273.0e0
        T[i + 1, j] = (depth) * dTdZ + offset

    elseif 110.0e3 > (depth) ≥ 35.0e3
        dTdZ = (1492 - 923) / 75.0e3
        offset = 923
        T[i + 1, j] = (depth - 35.0e3) * dTdZ + offset

    elseif (depth) ≥ 110.0e3
        dTdZ = (1837 - 1492) / 590.0e3
        offset = 1492.0e0
        T[i + 1, j] = (depth - 110.0e3) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            depth = abs(y[j])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i + 1, j] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end
    nx, ny = size(T)
    @parallel (1:(nx - 2), 1:ny) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ly = 700.0e3            # domain length in y
    lx = ly * ar          # domain length in x
    ni = nx, ny           # number of cells
    li = lx, ly           # domain length in x- and y-
    di = @. li / ni       # grid step in x- and -y
    origin = 0.0, -ly         # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; is_plastic = true)
    κ = (10 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Weno model -----------------------------------------
    weno = WENO5(backend_JR, Val(2), ni .+ 1)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 1
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi...)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(3))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2   # origin of thermal anomaly
    yc_anomaly = -610.0e3 # origin of thermal anomaly
    r_anomaly = 25.0e3   # radius of perturbation
    init_phases!(pPhases, particles, lx; d = abs(yc_anomaly), r = r_anomaly)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
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
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2])
    thermal_bcs!(thermal, thermal_bc)

    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        compute_ρg!(ρg[2], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    args = (; T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
    # Rheology
    viscosity_cutoff = (1.0e16, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    (; η, η_vep) = stokes.viscosity

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 1 / √2.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

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

    # WENO arrays
    T_WENO = @zeros(ni .+ 1)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    # Time loop
    t, it = 0.0, 0
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 5 # run only for 5 Myrs

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = dt, ΔTc = thermal.ΔTc)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, viscosity_cutoff
        )
        # ------------------------------

        # Stokes solver ----------------
        iters = solve!(
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
                iterMax = 150.0e3,
                nout = 1.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Weno advection
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
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            ),
        )
        T_WENO .= thermal.T[2:(end - 1), :]
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        WENO_advection!(T_WENO, (Vx_v, Vy_v), weno, di, dt)
        thermal.T[2:(end - 1), :] .= T_WENO
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
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
            # Make Makie figure
            fig = Figure(size = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1])
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3, Array(T_WENO), colormap = :batlow)
            Colorbar(fig[1, 2], h1)
            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end
        # ------------------------------

    end

    return iters
end
## END OF MAIN SCRIPT ----------------------------------------------------------------


# (Path)/folder where output data and figures are stored
figdir = "Weno2D"
ar = 1 # aspect ratio
n = 64
nx = n * ar
ny = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny);
