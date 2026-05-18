const isCUDA = false
# const isCUDA = true

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
using GeoParams
using GLMakie

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
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
    @all(P) = abs(@all(ρg) * (@all_j(z) + 50e3)) * <(@all_j(z), -50e3)
    return nothing
end

function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, index)
        
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue
            @index phases[ip, i, j] = 1.0
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.index)
end

@parallel_indices (i, j) function init_T!(T, x, y)
    if iszero(y[j]) && (8.5e-2 ≤ x[i] ≤ 11.5e-2)
        T[i + 1, j] = 273.0e0 + 60
    else
        T[i + 1, j] = 273.0e0 + 20
    end
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny)

    # Physical domain ------------------------------------
    lx = 20e-2             # domain length in x
    ly = 40e-2             # domain length in y
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, 0e0      # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    # r = HerschelBulkley(;
    #     σ0 = 0.05,
    #     Kv = 2e0,
    #     n = 0.7,
    # )
    r = HerschelBulkley(;
            n = 3.0,
            η0 = 1.0e24,
            τ0 = 100.0e6,
            ηr = 1.0e20,
            Q = 0.0,
            Tr = 1273,
        )
    rheology = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase             = 1,
            Density           = T_Density(; ρ0 = 1.144e3, T0=273.15, α = 6e-4),
            CompositeRheology = CompositeRheology((HerschelBulkley(),)),
            Conductivity      = ConstantConductivity(; k = 0.4),
            HeatCapacity      = ConstantHeatCapacity(; Cp = 800),
            Gravity           = ConstantGravity(; g = 9.81),
        ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 50, 75, 25
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    init_phases!(pPhases, particles)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(backend, ni)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(backend, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    @parallel (@idx ni .+ 1) init_T!(thermal.T, xvi...)
    thermal_bcs!(thermal, thermal_bc)
    Tbot = thermal.T[:, 1]
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
    compute_ρg!(ρg, phase_ratios, rheology, (T = thermal.Tc, P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])

    η_cutoff = (0.01, 1e8)
    compute_viscosity!(stokes, phase_ratios, args, rheology, η_cutoff)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip   = (left = false, right = false, top = false, bot = true),
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    figdir = "FatPlume"
    take(figdir)

    # PT coefficients for thermal diffusion
    dt    =  1
    pt_thermal = PTThermalCoeffs(
        backend, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-4, CFL = 1.0e-2 / √2.1
    )

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, T_buffer, particles)

    vtk_dir = joinpath(figdir, "vtk")
    take(vtk_dir)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # Time loop
    t, it = 0.0, 0
    dt    =  120
    # dtmax = 500.0e3 * (3600 * 24 * 365.25)
    while it < 500

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, particles)
        @views T_buffer[:, end] .= 273.0 + 20
        @views thermal.T[2:(end - 1), :] .= T_buffer
        @views thermal.T[:, 1] .= Tbot
        temperature2center!(thermal)

        # Stokes -----------------------
        iters = solve_DYREL!(
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
                verbose_DR = true,
                iterMax = 50.0e3,
                nout = 1,
                rel_drop = 0.1,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1e-4,
                linear_viscosity = true,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        dt = compute_dt(stokes, di) * 0.95
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
                iterMax = 50.0e3,
                nout = 1.0e3,
                verbose = true,
            ),
        )
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀,  dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:(end - 1), :], subgrid_arrays, particles, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,))
        # ------------------------------

        @show it += 1
        t += dt

        if it == 1 || rem(it, 10) == 0
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)

            (; η_vep, η) = stokes.viscosity
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.τ)

            # if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(T_buffer),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    Vx = Array(Vx_v),
                    Vy = Array(Vy_v),
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
                    velocity_v;
                    t = t
                )
            # end

            nt = 5
            fig = Figure(size = (900, 900), title = "t = $t")
            ax = Axis(fig[1, 1], aspect = 1, title = " t=$(t) s")

            # # Make particles plottable
            # p = particles.coords
            # ppx, ppy = p
            # pxv = ppx.data[:]
            # pyv = ppy.data[:]
            # clr = pPhases.data[:]
            # idxv = particles.index.data[:]

            # scatter!(ax, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), markersize = 5)
            # heatmap!(ax, xci..., log10.(stokes.viscosity.η_vep), colormap=:lipari)
            heatmap!(ax, xci..., thermal.Tc, colormap=:lipari)
            arrows!(
                ax,
                xvi[1][1:nt:(end - 1)], 
                xvi[2][1:nt:(end - 1)], 
                Array.((Vx_v[1:nt:(end - 1), 1:nt:(end - 1)], 
                Vy_v[1:nt:(end - 1), 1:nt:(end - 1)]))...,
                lengthscale =  1e-2 / max(maximum(Vx_v), maximum(Vy_v)),
                # lengthscale = 2e11,
                color = :white,
            )

            fig
            save(joinpath(figdir, "$(it).png"), fig)
        end
    end

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n  = 64
nx = n
ny = n * 2
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# main(igg, nx, ny)
