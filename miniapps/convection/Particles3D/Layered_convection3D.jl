using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, GeoParams, CairoMakie, GeoParams

# Load file with all the rheology configurations
include("Layered_rheology.jl")

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

# Initial thermal profile
@parallel_indices (I...) function init_T!(T, z)
    depth = -z[I[3]]

    if depth < 0.0e0
        T[I...] = 273.0

    elseif 0.0e0 ≤ (depth) < 35.0e3
        dTdZ = (923 - 273) / 35.0e3
        offset = 273.0e0
        T[I...] = (depth) * dTdZ + offset

    elseif 110.0e3 > (depth) ≥ 35.0e3
        dTdZ = (1492 - 923) / 75.0e3
        offset = 923
        T[I...] = (depth - 35.0e3) * dTdZ + offset

    elseif (depth) ≥ 110.0e3
        dTdZ = (1837 - 1492) / 590.0e3
        offset = 1492.0e0
        T[I...] = (depth - 110.0e3) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _rectangular_perturbation!(T, xc, yc, zc, r, x, y, z)
        @inbounds if (abs(x[i] - xc) ≤ r) && (abs(y[j] - yc) ≤ r) && (abs(z[k] - zc) ≤ r)
            depth = abs(z[k])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i, j, k] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end

    return @parallel _rectangular_perturbation!(T, xc, yc, zc, r, xvi...)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main3D(igg; ar = 1, nx = 16, ny = 16, nz = 16, figdir = "figs3D", do_vtk = false)

    # Physical domain ------------------------------------
    lz = 700.0e3                # domain length in z
    lx = ly = lz * ar              # domain length in x and y
    ni = nx, ny, nz           # number of cells
    li = lx, ly, lz           # domain length
    di = @. li / ni           # grid steps
    origin = 0.0, 0.0, -lz        # origin coordinates (15km of sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; is_plastic = true)
    κ = (10 / (rheology[1].HeatCapacity[1].Cp * rheology[1].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^3 / κ / 3.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 25
    max_xcell = 35
    min_xcell = 8
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni)

    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2   # origin of thermal anomaly
    yc_anomaly = ly / 2   # origin of thermal anomaly
    zc_anomaly = -610.0e3 # origin of thermal anomaly
    r_anomaly = 50.0e3   # radius of perturbation
    init_phases!(pPhases, particles, lx, ly; d = abs(zc_anomaly), r = r_anomaly)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-4, CFL = 0.5 / √3.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )
    # initialize thermal profile - Half space cooling
    @parallel init_T!(thermal.T, xvi[3])
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, zc_anomaly, r_anomaly, xvi)
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[end], xci[end])

    # Rheology
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e18, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 1.0e-3 / √3
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
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

    # Plot initial T and η profiles
    fig = let
        Zv = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
        Z = [z for x in xci[1], y in xci[2], z in xci[3]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[:]), Zv ./ 1.0e3)
        lines!(ax2, Array(log10.(stokes.viscosity.η[:])), Z ./ 1.0e3)
        ylims!(ax1, minimum(xvi[3]) ./ 1.0e3, 0)
        ylims!(ax2, minimum(xvi[3]) ./ 1.0e3, 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    grid2particle!(pT, xvi, thermal.T, particles)
    dt₀ = similar(stokes.P)

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end
    # Time loop
    t, it = 0.0, 0
    while (t / (1.0e6 * 3600 * 24 * 365.25)) < 5 # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(thermal.T, pT, xvi, particles)
        temperature2center!(thermal)

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_ρg!(ρg[end], phase_ratios, rheology, args)
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, viscosity_cutoff
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
            Inf,
            igg;
            kwargs = (;
                iterMax = 100.0e3,
                nout = 1.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff) / 2
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
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
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

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy, grid_vz), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (thermal.T,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(thermal.T),
                    τxy = Array(stokes.τ.xy),
                    εxy = Array(stokes.ε.xy),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
                    Vz = Array(Vz_v),
                )
                data_c = (;
                    Tc = Array(thermal.Tc),
                    P = Array(stokes.P),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    η = Array(log10.(stokes.viscosity.η_vep)),
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
                    velocity_v,
                    t = t
                )
            end

            slice_j = ny >>> 1
            # Make Makie figure
            fig = Figure(size = (1400, 1800), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "τII [MPa]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xvi[1] .* 1.0e-3, xvi[3] .* 1.0e-3, Array(thermal.T[:, slice_j, :]), colormap = :lajolla)
            # Plot particles phase
            h2 = heatmap!(ax2, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(stokes.τ.II[:, slice_j, :] .* 1.0e-6), colormap = :batlow)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.ε.II[:, slice_j, :])), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1] .* 1.0e-3, xci[3] .* 1.0e-3, Array(log10.(stokes.viscosity.η_vep[:, slice_j, :])), colormap = :batlow)
            hideydecorations!(ax3)
            hideydecorations!(ax4)
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
## END OF MAIN SCRIPT ----------------------------------------------------------------

do_vtk = true # set to true to generate VTK files for ParaView
ar = 1 # aspect ratio
n = 32
nx = n
ny = n
nz = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

# (Path)/folder where output data and figures are stored
figdir = "Plume3D_$n"
main3D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk);
