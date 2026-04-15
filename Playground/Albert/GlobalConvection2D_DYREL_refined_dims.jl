# const isGPU = true 
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
include("GlobalConvectionrheology_dims.jl")

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
function init_T!(T, y, thick_air)

    ni = size(T) .- (2, 0)

    d_air = thick_air
    d_0km = 0.0e0
    d_35km = 35.0e3
    d_110km = 110.0e3

    T_273K = 293.0e0
    T_18K = (923 - 273) / 35.0e3
    T_7K = (1492 - 923) / 75.0e3
    T_0_5K = (1837 - 1492) / 590.0e3
    T_923K = 923.0e0
    T_1492K = 1492.0e0

    @parallel_indices (i, j) function init_T2!(T, y)
        depth = -y[j] - d_air

        # (depth - 15e3) because we have 15km of sticky air
        if depth < d_0km
            T[i + 1, j] = T_273K

        elseif d_0km ≤ depth < d_35km
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
            T[i + 1, j] = 0 * (depth - d_110km) * dTdZ + offset
        end

        return nothing
    end

    @parallel (@idx ni) init_T2!(T, y)

    return nothing
end


# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi, thick_air)

    dTdZ = (2047 - 2017) / 50.0e3
    offset = 2017.0e0
    d_585km = 585.0e3
    d_air = thick_air

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc - d_air)^2 ≤ r^2)
            depth = -y[j] - d_air
            T[i + 1, j] = (depth - d_585km) * dTdZ + offset
        end
        return nothing
    end

    nx, ny = size(T)
    @parallel (1:(nx - 2), 1:ny) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end

# α  --> outer plateau height
# d1 --> half-width of dip (top)
# d2 --> half-width of dip (bottom)
# w  --> sharpness
# c --> shift of center
function tanh_monitor2(α, c, d1, d2, w)
    x ->( α + 0.5*α * (tanh((x - (c + d1))/w) - tanh((x - (c - d2))/w))) + 1
end

## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", do_vtk = false)

    thickness = 2890.0e3
    # Physical domain ------------------------------------
    thick_air = 0.0e0                 # thickness of sticky air layer
    ly = thickness + thick_air # domain length in y
    lx = ly * ar           # domain length in x
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, -ly      # origin coordinates (15km f sticky air layer)
    grid0 = Geometry(ni, li; origin = origin)
    # α, κ, c = 5, 50, -0.2
    # M = tanh_monitor(α, κ, c; direction = :right)

    # α  --> outer plateau height
    # d1 --> half-width of dip (top)
    # d2 --> half-width of dip (bottom)
    # w  --> sharpness
    # c --> shift of center
    # α, c, d1, d2, w = 5, -0.5 * thickness, 0.4 * thickness, 0.45 * thickness, 1e-2 * thickness
    # M = tanh_monitor2(α, c, d1, d2, w)
    # xv_ref = solve_grid(grid0.xvi[2][1], grid0.xvi[2][end], M, ny) # refined grid
    grid = Geometry(
        PTArray(backend_JR),
        collect(grid0.xvi[1]),
        collect(grid0.xvi[2]),
        # xv_ref,
    )
    di_min = minimum.(grid.di.vertex)

    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies(; is_plastic = true)
    κ = (4 / (rheology[4].HeatCapacity[1].Cp * rheology[4].Density[1].ρ0))
    dt = 0.5 * min(di_min...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 50, 75, 20
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
    yc_anomaly = -2800.0e3 # origin of thermal anomaly
    r_anomaly = 150.0e3 # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, lx, yc_anomaly, r_anomaly, thick_air)
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
    # init_T!(thermal.T, xvi[2], thick_air)
    # rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi, thick_air)
    # dTdz = (3273.0e0 - 293.0e0) / grid.li[2]
    # thermal.T[2:end-1, :]  .= PTArray(backend_JR)([ -dTdz * y + 293.0e0 for x in Array(grid.xvi[1]), y in Array(grid.xvi[2]) ]    )
    # thermal.T[2:end-1, :] .+= PTArray(backend_JR)([A * sin(π*x/grid.li[1]) * sin(π*y/grid.li[1]) for x in Array(grid.xvi[1]), y in Array(grid.xvi[2]) ])
    thermal.T[2:end-1, :]  .= PTArray(backend_JR)([
        T_field(
                x / 1.0e3, 
                -z / 1.0e3; 
                Lx      = grid.li[1] / 1.0e3,
                Lz      = grid.li[2] / 1.0e3, 
                A       = 150.0,      # perturbation amplitude in K
                Ttop    = 273.0,
                Tbot    = 2800.0,
                Tm      = 1350.0,
                delta_b = 250.0,
                w_t     = 2.5,
                w_b     = 5
        )
        for x in Array(grid.xvi[1]), z in Array(grid.xvi[2])
    ])
    # thermal.T[2:end-1, :]  .*= @rand(ni.+1...) .* 0.05
    thermal_bcs!(thermal, thermal_bc)
    Ttop, Tbot = extrema(thermal.T)
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
    viscosity_cutoff = (1.0e18, 1.0e23)
    stokes.ε.xx .= 1.0e-15
    stokes.ε.xx_v .= 1.0e-15
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
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
        Y = [y for x in Array(xci[1]), y in Array(xci[2])][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        scatter!(
            ax1, 
            Array(thermal.T[2:(end - 1), :][:]), 
            Yv./1e3
        )
        scatter!(
            ax2, 
            log10.(Array(stokes.viscosity.η[:])), 
            Y./1e3
        )
        # ylims!(ax1, minimum(xvi[2]), 0)
        # ylims!(ax2, minimum(xvi[2]), 0)
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

    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1e-4)

    while t < 10.0e6 * 365.25 * 24 * 3600

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
                verbose_PH           = true,
                verbose_DR           = false,
                iterMax              = 50.0e3,
                nout                 = 200,
                rel_drop             = 0.1,
                λ_relaxation_PH      = 1,
                λ_relaxation_DR      = 1,
                viscosity_relaxation = 1.0e-2,
                viscosity_cutoff     = viscosity_cutoff,
            )
        )
        tensor_invariant!(stokes.ε)
        dt = compute_dt(stokes, di_min) 
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
        advection!(particles, RungeKutta4(), @velocity(stokes), dt)
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
        
        t_dim = Float16(t / (365.25 * 24 * 3600) / 1.0e3)
        dt_dim = Float16(dt / (365.25 * 24 * 3600) / 1.0e3)
        println("\n\n
            t  = $t_dim [kyr]
            dt = $dt_dim [kyr]\n\n"
        )
        
        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, particles)
        @views thermal.T[2:(end - 1), :] .= T_buffer
        @views thermal.T[:, end] .= Ttop
        @views thermal.T[:, 1] .= Tbot
        thermal_bcs!(thermal, thermal_bc)
        temperature2center!(thermal)
        # ------------------------------

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0
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
                    τII = Array(stokes.τ.II),
                    εxx = Array(stokes.ε.xx),
                    εyy = Array(stokes.ε.yy),
                    εII = Array(stokes.ε.II),
                    η = Array(stokes.viscosity.η_vep),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
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
            t_dim = Float16(t / (365.25 * 24 * 3600) / 1.0e6)
            fig = Figure(size = (1600, 800).*2, title = "t = $t_dim [Myr]")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K] ; t=$t_dim [Myr]")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "phase")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "log10(η)")
            # Plot temperature
            h1 = heatmap!(ax1, Array.(xvi)..., Array(thermal.T[2:(end - 1), :]), colormap = :glasgow)
            # Plot particles phase
            h2 = scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize=1)
            # Plot 2nd invariant of strain rate
            h3 = heatmap!(ax3, Array.(xci)..., Array(log10.(stokes.ε.II)), colormap = :bamO)
            # Plot effective viscosity
            h4 = heatmap!(ax4, Array.(xci)..., Array(log10.(stokes.viscosity.η)), colormap = :lipari)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1, height = 3e2)
            Colorbar(fig[2, 2], h2, height = 3e2)
            Colorbar(fig[1, 4], h3, height = 3e2)
            Colorbar(fig[2, 4], h4, height = 3e2)
            linkaxes!(ax1, ax2, ax3, ax4)
            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

ar = 4    # aspect ratio
n  = 128 ÷ 1
nx = n * ar
ny = n

# (Path)/folder where output data and figures are stored
figdir = "Convection2D_nx$(nx)"
do_vtk = true # set to true to generate VTK files for ParaView

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
# main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, do_vtk = do_vtk);



# Adiff =  2.2e-10Pa^(-1) * s^(-1)
# creep = DiffusionCreep( # dry olivine
#     n = 1.0NoUnits,                         # power-law exponent
#     r = 0.0NoUnits,                         # exponent of water-fugacity
#     p = 0NoUnits,                           # grain size exponent
#     A = Adiff,    # material specific rheological parameter
#     E = 375.0e3J / mol,                      # activation energy
#     V = 3.65e-6m^3 / mol,                   # activation Volume
# )

# disl = DislocationCreep(
#     A = 1.1e-17Pa^(-35 // 10) / s,   # material specific rheological parameter
#     n = 3.5NoUnits,
#     E = 530.0e3J / mol,
#     V = 13.0e-6m^3 / mol,
#     r = 0.0NoUnits,
#     R = 8.3145J / mol / K,
# )

# diff = DiffusionCreep(
#     n = 1.0NoUnits,                         # power-law exponent
#     r = 0.0NoUnits,                         # exponent of water-fugacity
#     p = 0NoUnits,                           # grain size exponent
#     A = Adiff,        # material specific rheological parameter
#     E = 375e3J / mol,                      # activation energy
#     V = 3.65e-6m^3 / mol,                   # activation Volume
# )

# T=thermal.T[2, :][:]
# P=stokes.P[1,:][:]

# P[end-42]
# T[end-42]

# εII = 1e-15
# dt = 4
# # P = 8.8e9
# # T = 1150 + 273.15
# args = (; P = P[end-42], T = T[end-42], dt = dt)

# ηdisl  = compute_viscosity_εII(disl, εII, args)
# ηdiff  = compute_viscosity_εII(diff, εII, args)

# # Rheology
# args = (; P = stokes.P, T = thermal.Tc, dt = Inf);
# compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; relaxation=1)

# dimensionalize(165, Pa*s, CharDim)

# ratio_ij, AII, fn_viscosity, args_ij = ([0.0, 0.0, 0.0, 1.0, 0.0], 1.0e-15, GeoParams.compute_viscosity_εII, (P = 3.1172135391023838e10, T = 1654.4272264125677, dt = Inf, τII_old = 0.0))

# # @edit JustRelax2D.compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)
# eta = JustRelax2D.compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)

# eta = JustRelax2D.compute_phase_viscosity(rheology, ratio_ij, AII, fn_viscosity, args_ij)

# compute_viscosity_εII(rheology[4].CompositeRheology[1], 1e-15, args_ij)
# @edit compute_viscosity_εII(rheology[4].CompositeRheology[1], 1e-15, args)
