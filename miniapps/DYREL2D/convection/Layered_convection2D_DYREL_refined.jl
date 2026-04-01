const isGPU = false

@static if isGPU
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO

@static if isGPU
    const backend_JR = CUDABackend
else
    const backend_JR = CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@static if isGPU
    @init_parallel_stencil(CUDA, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.

const backend = @static if isGPU
    const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, GLMakie

using PoissonGrids

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
function init_T!(T, y, thick_air, CharDim)

    ni = size(T) .- (2,0)
    
    # non dim depths
    d_air = nondimensionalize(thick_air * km, CharDim)
    d_0km = nondimensionalize(0e0km, CharDim)
    d_35km = nondimensionalize(35e0km, CharDim)
    d_110km = nondimensionalize(110e0km, CharDim)

    # non dim T and gradients
    T_273K = nondimensionalize(293.0e0K, CharDim)
    T_18K = nondimensionalize((923 - 273) / 35 * K / km, CharDim)
    T_7K = nondimensionalize((1492 - 923) / 75 * K / km, CharDim)
    T_0_5K = nondimensionalize((1837 - 1492) / 590 * K / km, CharDim)
    T_923K = nondimensionalize(923.0e0K, CharDim)
    T_1492K = nondimensionalize(1492.0e0K, CharDim)

    @parallel_indices (i, j) function init_T2!(T, y)
        depth = -y[j] - d_air

        # (depth - 15e3) because we have 15km of sticky air
        if depth < d_0km
            T[i + 1, j] = T_273K

        elseif d_0km ≤ depth <  d_35km
            dTdZ = T_18K

            offset = T_273K
            T[i + 1, j] = (depth) * dTdZ + offset

        elseif d_110km > depth ≥ d_35km
            dTdZ = T_7K
            offset = T_923K
            T[i + 1, j] = (depth - d_35km) * dTdZ + offset

        elseif depth ≥ d_110km
            dTdZ = T_0_5K
            offset = T_1492K
            T[i + 1, j] = 0*(depth - d_110km) * dTdZ + offset
        end

        return nothing
    end

    @parallel (@idx ni) init_T2!(T, y)
    
    return nothing
end


# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi, thick_air, CharDim)

    dTdZ_nd   = nondimensionalize((2047 - 2017)K / 50km, CharDim)
    offset_nd = nondimensionalize(2017.0e0K, CharDim)
    d_585km   = nondimensionalize(585km, CharDim)
    ΔT        = nondimensionalize(100.0e0K, CharDim)
    d_air     = nondimensionalize(thick_air * km, CharDim)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc - d_air)^2 ≤ r^2)
            depth = -y[j] - d_air
            T[i + 1, j] = (depth - d_585km) * dTdZ_nd + offset_nd
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

    thickness = 700 * km
    η0 = 1.0e20
    CharDim = GEO_units(;
        length = thickness, viscosity = η0, temperature = 1.0e3K
    )
    # Physical domain ------------------------------------
    thick_air = nondimensionalize(0.0e0km, CharDim)                 # thickness of sticky air layer
    ly        = nondimensionalize(thickness, CharDim) + thick_air # domain length in y
    lx        = ly * ar           # domain length in x
    ni        = nx, ny            # number of cells
    li        = lx, ly            # domain length in x- and y-
    di        = @. li / ni        # grid step in x- and -y
    origin    = 0.0, -ly      # origin coordinates (15km f sticky air layer)
    grid0     = Geometry(ni, li; origin = origin)
    α, κ, c   = 5, 20, -0.1
    M         = tanh_monitor(α, κ, c; direction = :right)
    xv_ref    = solve_grid(grid0.xvi[2][1], grid0.xvi[2][end], M, ny) # refined grid
    grid      = Geometry(
        PTArray(backend_JR),
        collect(grid0.xvi[1]),
        # collect(grid0.xvi[2]),
        xv_ref,
    )
    di_min = min(
        min(minimum.(grid.di.center)...),
        min(minimum.(grid.di.vertex)...),
    )
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(CharDim; is_plastic = true)
    κ = (4 / (rheology[4].HeatCapacity[1].Cp * rheology[4].Density[1].ρ0))
    dt = 0.5 * di_min^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 25, 30, 8
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # particle fields for the stress rotation
    pτ = StressParticles(particles)
    particle_args = (pT, pPhases, unwrap(pτ)...)
    particle_args_reduced = (pT, unwrap(pτ)...)

    # Elliptical temperature anomaly
    xc_anomaly = lx / 2    # origin of thermal anomaly
    yc_anomaly = nondimensionalize(-610km, CharDim) # origin of thermal anomaly
    r_anomaly = nondimensionalize(25km, CharDim) # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, lx, yc_anomaly, r_anomaly, thick_air, CharDim)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    init_T!(thermal.T, xvi[2], thick_air, CharDim)
    thermal_bcs!(thermal, thermal_bc)
    Ttop, Tbot = extrema(thermal.T)
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi, thick_air, CharDim)
    temperature2center!(thermal)
    # ----------------------------------------------------
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:5
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        stokes.P .= PTArray(backend_JR)(reverse(cumsum(reverse(ρg[2] .* grid.di.vertex[2]', dims = 2), dims = 2), dims = 2))
    end

    # Rheology
    viscosity_cutoff = nondimensionalize((1.0e16Pa * s, 1.0e24Pa * s), CharDim)
    stokes.ε.II .= nondimensionalize(1.0e-8 / s, CharDim)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)
    # ------------------------------

    # PT coefficients for thermal diffusion
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, minimum.(grid.di.vertex), li; ϵ = 1.0e-6, CFL = 0.9 / √2.1
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
        Yv = [y for x in Array(xvi[1]), y in Array(xvi[2])][:]
        Y  = [y for x in Array(xci[1]), y in Array(xci[2])][:]
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
    grid2particle!(pT, T_buffer, particles)

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end

    # Time loop
    t, it = 0.0, 0

    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt)

    while t < nondimensionalize(5.0e6yr, CharDim) # run only for 5 Myrs

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, particles)
        @views thermal.T[2:(end - 1), :] .= T_buffer
        @views thermal.T[:, end] .= Ttop
        @views thermal.T[:, 1] .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        # ------------------------------

        solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 200,
                rel_drop = 0.1,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1.0e-3,
                viscosity_cutoff = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di_min) / 2
        # ------------------------------

        # rotate stresses
        rotate_stress!(pτ, stokes, particles, dt)
        # compute strain rate 2nd invartian - for plotting
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(
            particles,
            pPhases,
            particle_args_reduced,
            (T_buffer, stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.ω.xy),
        )
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        # interpolate stress back to the grid
        stress2grid!(stokes, pτ, particles)

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
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
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes
        )
        centroid2particle!(subgrid_arrays.dt₀, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:(end - 1), :], subgrid_arrays, particles, dt
        )
        # ------------------------------
      
        @show it += 1
        t += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0
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
                    Array.(xvi),
                    Array.(xci),
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
            h1 = heatmap!(ax1, Array.(xvi)..., Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))), colormap = :batlow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = :bilbao)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, Array.(xci)..., Array(log10.(ustrip.(dimensionalize(stokes.ε.II, s^-1, CharDim)))), colormap = :batlow)
            # Plot effective viscosity
            h4 = heatmap!(ax4, Array.(xci)..., Array(log10.(ustrip.(dimensionalize(stokes.viscosity.η_vep, Pa * s, CharDim)))), colormap = :batlow)
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

ar = 1 # aspect ratio
n = 64
nx = n * ar
ny = n 

# (Path)/folder where output data and figures are stored
figdir = "Plume2D_x$(n)_refined"
do_vtk = true # set to true to generate VTK files for ParaView

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
# main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, do_vtk = do_vtk);