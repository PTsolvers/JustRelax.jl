const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")

const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

# The array backend and the ParallelStencil backend have to be the same one: `@zeros` allocates
# through ParallelStencil, and those arrays are handed to kernels alongside the `StokesArrays`.
@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using LinearAlgebra, GeoParams, CairoMakie

# Velocity helper grids for the particle advection
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

function init_phases!(phases, particles, A)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, A)

        f(x, A, λ) = A * sin(π * x / λ)

        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            @index phases[ip, i, j] = 2.0

            if 0.0e0 ≤ depth ≤ 100.0e3
                @index phases[ip, i, j] = 1.0

            elseif depth > (-f(x, A, 500.0e3) + (200.0e3 - A))
                @index phases[ip, i, j] = 3.0

            end

        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, A)
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(igg, nx, ny; figdir = "RayleighTaylor2D_DYREL")

    # Physical domain ------------------------------------
    thick_air = 100.0e3             # thickness of sticky air layer
    ly = 500.0e3 + thick_air # domain length in y
    lx = 500.0e3             # domain length in x
    ni = nx, ny            # number of cells
    li = lx, ly            # domain length in x- and y-
    di = @. li / ni        # grid step in x- and -y
    origin = 0.0, -ly          # origin coordinates (15km f sticky air layer)
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = rheology = (
        # Name              = "Air",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e16),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Crust",
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Mantle",
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 60, 80, 40
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    # temperature
    pT, pPhases = init_cell_arrays(particles, Val(2))
    particle_args = (pT, pPhases)

    # Elliptical temperature anomaly
    A = 5.0e3    # Amplitude of the anomaly
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, A)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-4, Re = 3.0e0, r = 0.7, CFL = 0.98 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    # ----------------------------------------------------

    # Buoyancy forces & rheology
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = thermal.T, P = stokes.P, dt = Inf)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
        free_surface = true,
    )

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    take(figdir)

    # Time loop
    t, it = 0.0, 0
    dt = 10.0e3 * (3600 * 24 * 365.25)
    dt_max = 25.0e3 * (3600 * 24 * 365.25)

    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, γfact = 110)

    while it < 20 # 100

        ## variational solver
        # Stokes solver ----------------
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
                iterMax = 250.0e3,
                nout = 100,
                rel_drop = 0.75,
                λ_relaxation_PH = 1,
                λ_relaxation_DR = 1,
                viscosity_relaxation = 1.0e-2,
                linear_viscosity = true,
                free_surface = true,
                verbose_PH = true,
                verbose_DR = false,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        dt = compute_dt(stokes, di, dt_max)
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), ())
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        @show it += 1
        t += dt

        if it == 1 || rem(it, 10) == 0
            px, py = particles.coords
          
            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            nt = 10
            println("Plotting...")
            # One marker per particle costs seconds per figure and dominates the time loop once
            # the solve runs on a GPU. The phase field goes in as a heatmap of the dominant
            # phase per cell, and only every `nskip`-th particle is drawn over it.
            nskip = 4
            # `.data` is laid out as (cells, slots), so the slot axis is the one to stride:
            # striding the flattened array drops whole columns of cells and stripes the figure.
            sub_particles(A) = vec(Array(A.data)[:, 1:nskip:end, :])
            phase_c = Array([argmax(p) for p in Array(phase_ratios.center)])
            fig = Figure(size = (900, 900), title = "t = $t")
            ax = Axis(fig[1, 1], aspect = 1, title = " t=$(round.(t / (1.0e3 * 3600 * 24 * 365.25); digits = 3)) Kyrs")
            # heatmap!(ax, xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, phase_c, colormap = :grayC)
            scatter!(
                ax, sub_particles(px) .* 1.0e-3, sub_particles(py) .* 1.0e-3,
                color = sub_particles(pPhases), 
                colormap = :vikO, 
                # colormap = :grayC, 
                markersize = 3,
            )
            arrows2d!(
                ax,
                xvi[1][1:nt:(end - 1)] ./ 1.0e3, xvi[2][1:nt:(end - 1)] ./ 1.0e3, Array.((Vx_v[1:nt:(end - 1), 1:nt:(end - 1)], Vy_v[1:nt:(end - 1), 1:nt:(end - 1)]))...,
                lengthscale = 25 / max(maximum(Vx_v), maximum(Vy_v)),
                color = :white,
            )
            fig
            save(joinpath(figdir, "$(it).png"), fig)
            println("... figure saved.")

        end
    end
    return nothing
end

## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
n = 256
nx = n
ny = n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

main(igg, nx, ny; figdir = "RayleighTaylor2D_DYREL")