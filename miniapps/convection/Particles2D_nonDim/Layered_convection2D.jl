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
using GeoParams, GLMakie

# Load file with all the rheology configurations
include("Layered_rheology.jl")

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

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y, thick_air, CharDim)
    depth = -y[j] - thick_air

    # (depth - 15e3) because we have 15km of sticky air
    if depth < nondimensionalize(0.0e0km, CharDim)
        T[i + 1, j] = nondimensionalize(273.0e0K, CharDim)

    elseif nondimensionalize(0.0e0km, CharDim) ≤ (depth) < nondimensionalize(35km, CharDim)
        dTdZ = nondimensionalize((923 - 273) / 35 * K / km, CharDim)

        offset = nondimensionalize(273.0e0K, CharDim)
        T[i + 1, j] = (depth) * dTdZ + offset

    elseif nondimensionalize(110km, CharDim) > (depth) ≥ nondimensionalize(35km, CharDim)
        dTdZ = nondimensionalize((1492 - 923) / 75 * K / km, CharDim)
        offset = nondimensionalize(923K, CharDim)
        T[i + 1, j] = (depth - nondimensionalize(35km, CharDim)) * dTdZ + offset

    elseif (depth) ≥ nondimensionalize(110km, CharDim)
        dTdZ = nondimensionalize((1837 - 1492) / 590 * K / km, CharDim)
        offset = nondimensionalize(1492.0e0K, CharDim)
        T[i + 1, j] = (depth - nondimensionalize(110km, CharDim)) * dTdZ + offset

    end

    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi, thick_air, CharDim)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, CharDim, x, y)
        if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc - thick_air)^2 ≤ r^2)
            depth = -y[j] - thick_air
            dTdZ = nondimensionalize((2047 - 2017)K / 50km, CharDim)
            offset = nondimensionalize(2017.0e0K, CharDim)
            T[i + 1, j] = (depth - nondimensionalize(585km, CharDim)) * dTdZ + offset
        end
        return nothing
    end

    nx, ny = size(T)
    @parallel (1:(nx - 2), 1:ny) _rectangular_perturbation!(T, xc, yc, r, CharDim, xvi...)

    return nothing
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", do_vtk = false)

    thickness = 700 * km
    η0 = 1.0e20
    CharDim = GEO_units(;
        length = thickness, viscosity = η0, temperature = 1.0e3K
    )
    # Physical domain ------------------------------------
    thick_air = nondimensionalize(0.0e0km, CharDim)                 # thickness of sticky air layer
    ly = nondimensionalize(thickness, CharDim) + thick_air # domain length in y
    lx = ly * ar           # domain length in x
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim; is_plastic = true)
    κ = (4 / (rheology[4].HeatCapacity[1].Cp * rheology[4].Density[1].ρ0))
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 30, 8
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2    # origin of thermal anomaly
    yc_anomaly = nondimensionalize(-610km, CharDim) # origin of thermal anomaly
    r_anomaly = nondimensionalize(25km, CharDim) # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, lx, yc_anomaly, r_anomaly, thick_air, CharDim)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-7, CFL = 0.9 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi[2], thick_air, CharDim)
    thermal_bcs!(thermal, thermal_bc)
    Tbot = thermal.T[1, 1]
    Ttop = thermal.T[1, end]
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi, thick_air, CharDim)
    temperature2center!(thermal)
    # ----------------------------------------------------
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:1
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    viscosity_cutoff = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-6, CFL = 1.0e-3 / √2.1
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
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
        scatter!(ax1, Array(thermal.T[2:(end - 1), :][:]), Yv)
        scatter!(ax2, Array(log10.(stokes.viscosity.η[:])), Y)
        ylims!(ax1, minimum(xvi[2]), 0)
        ylims!(ax2, minimum(xvi[2]), 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end
    # Time loop
    t, it = 0.0, 0
    while t < nondimensionalize(5.0e6yr, CharDim) # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views thermal.T[2:(end - 1), :] .= T_buffer
        @views thermal.T[:, end] .= Ttop
        @views thermal.T[:, 1] .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_ρg!(ρg[end], phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
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
                iterMax = 150.0e3,
                nout = 1.0e3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di, dt_diff)
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
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            ),
        )
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:(end - 1), :], subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 25) == 0
            checkpointing_hdf5(figdir, stokes, thermal.T, t, dt)

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))),
                    τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CharDim))),
                    εxy = Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CharDim))),
                    Vx = Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                    Vy = Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
                )
                data_c = (;
                    P = Array(ustrip.(dimensionalize(stokes.P, MPa, CharDim))),
                    τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa, CharDim))),
                    τyy = Array(ustrip.(dimensionalize(stokes.τ.yy, MPa, CharDim))),
                    τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CharDim))),
                    εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1, CharDim))),
                    εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1, CharDim))),
                    εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1, CharDim))),
                    η = Array(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CharDim))),
                )
                velocity_v = (
                    Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                    Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
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
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = pPhases.data[:]
            idxv = particles.index.data[:]

            # Make Makie figure
            t_dim = Float16(dimensionalize(t, yr, CharDim).val / 1.0e3)
            fig = Figure(size = (900, 900), title = "t = $t_dim [kyr]")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K] ; t=$t_dim [kyrs]")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "phase")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))), colormap = :batlow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = :grayC)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, xci[1], xci[2], Array(log10.(ustrip.(dimensionalize(stokes.ε.II, s^-1, CharDim)))), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CharDim)))), colormap = :batlow)
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
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
figdir = "Plume2D"
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

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, do_vtk = do_vtk);
