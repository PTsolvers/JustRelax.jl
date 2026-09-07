const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")

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

using JustPIC
const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

# Load script dependencies
using GeoParams, CairoMakie

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Thermal rectangular perturbation

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, xc, yc, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j])
            # plume - rectangular
            @index phases[ip, i, j] = if ((x - xc)^2 ≤ r^2) && ((depth - yc)^2 ≤ r^2)
                2.0
            else
                1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, xc, yc, r)
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    return esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg) * abs(@all_j(z))
    return nothing
end

# --------------------------------------------------------------------------------
# BEGIN MAIN SCRIPT
# --------------------------------------------------------------------------------
function sinking_block2D(igg; ar = 8, ny = 16, nx = ny * 8, figdir = "figs2D", thermal_perturbation = :circular)

    # Physical domain ------------------------------------
    ly = 500.0e3
    lx = ly * ar
    origin = -lx / 2, -ly                         # origin coordinates
    ni = nx, ny                           # number of cells
    li = lx, ly                           # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    δρ = 100
    rheology = (
        SetMaterialParams(;
            Name = "Mantle",
            Phase = 1,
            Density = ConstantDensity(; ρ = 3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        SetMaterialParams(;
            Name = "Block",
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.2e3 + δρ),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
    # heat diffusivity
    dt = 1
    # ----------------------------------------------------

    grid_vxi = velocity_grids(xci, xvi, di)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 40, 40, 12
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    # temperature
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    # Rectangular density anomaly
    xc_anomaly = 0.0      # centred horizontally (domain spans -lx/2 .. lx/2)
    yc_anomaly = -ly / 2  # centred vertically; `init_phases!` takes it as a depth
    r_anomaly = 50.0e3   # radius of perturbation
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, xc_anomaly, abs(yc_anomaly), r_anomaly)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Adjoint Stokes --------------------------------------
    stokes_ad = AdjointStokesArrays(backend, ni)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = @ones(ni .+ 2...), P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt = dt)
    viscosity_cutoff = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    # ----------------------------------------------------

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    it = 0 # iteration counter
    while it < 1
        # Stokes solver ----------------
        args = (; T = @ones(ni .+ 2...), P = stokes.P, dt = dt, ΔT = @zeros(ni .+ 2...))
        solve_DYREL!(
            stokes,
            stokes_ad,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            verbose_PH = true,
            verbose_DR = false,
            iterMax = 50.0e3,
            nout = 10,
            rel_drop = 1.0e-2,
            λ_relaxation_PH = 1,
            λ_relaxation_DR = 1,
            viscosity_relaxation = 1,
            viscosity_cutoff = viscosity_cutoff,
            adjoint = true,
            observation = (;
                field = :Vy,
                center = (xc_anomaly, yc_anomaly),
                half_width = (r_anomaly, r_anomaly),
            ),
        )
        dt = compute_dt(stokes, di, igg) * 0.8
        # ------------------------------

        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        velocity = @. √(Vx_v^2 + Vy_v^2)

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        # advect particles in memory
        move_particles!(particles, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), ())
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)


        # Plotting ---------------------
        # adjoint velocities live on the same staggered grid as the forward ones,
        # so they get interpolated to the vertices the same way
        λVx_v = @zeros(ni .+ 1...)
        λVy_v = @zeros(ni .+ 1...)
        velocity2vertex!(λVx_v, λVy_v, stokes_ad.λV.Vx, stokes_ad.λV.Vy)

        xc_km = xci[1] .* 1.0e-3, xci[2] .* 1.0e-3   # cell centres
        xv_km = xvi[1] .* 1.0e-3, xvi[2] .* 1.0e-3   # vertices
        ρ = Array(ρg[2]) ./ 9.81

        # a constant field (all-zero adjoint arrays, say) has no colour range Makie can use
        function crange(A, diverging)
            if diverging
                m = maximum(abs, A)
                return iszero(m) ? (-1.0, 1.0) : (-m, m)
            end
            lo, hi = extrema(A)
            return lo == hi ? (lo - 1, hi + 1) : (lo, hi)
        end

        f = Figure(size = (1900, 950))
        function panel!(row, col, title, x, A; colormap = :batlow, diverging = false)
            ax = Axis(
                f[row, 2col - 1];
                aspect = DataAspect(), title = title,
                xlabel = row == 2 ? "x [km]" : "", ylabel = col == 1 ? "y [km]" : "",
            )
            h = heatmap!(ax, x..., Array(A); colormap = colormap, colorrange = crange(Array(A), diverging))
            Colorbar(f[row, 2col], h)
            row == 1 && hidexdecorations!(ax, grid = false)
            col > 1 && hideydecorations!(ax, grid = false)
            return nothing
        end

        # forward Stokes solution
        panel!(1, 1, "ρ [kg/m³]", xc_km, ρ; colormap = :roma)
        panel!(1, 2, "Vx [m/s]", xv_km, Vx_v; colormap = :vik, diverging = true)
        panel!(1, 3, "Vy [m/s]", xv_km, Vy_v; colormap = :vik, diverging = true)
        panel!(1, 4, "P [Pa]", xc_km, stokes.P; colormap = :batlow)

        # adjoint solution
        panel!(2, 1, "λVx", xv_km, λVx_v; colormap = :vik, diverging = true)
        panel!(2, 2, "λVy", xv_km, λVy_v; colormap = :vik, diverging = true)
        panel!(2, 3, "ρ (adjoint)", xc_km, stokes_ad.ρ; colormap = :roma)
        panel!(2, 4, "η (adjoint)", xc_km, stokes_ad.η; colormap = :batlow)

        save(
            joinpath(@__DIR__, "sinking_DR_$(it).png"),
            f
        )
        display(f)
        it += 1
        @show extrema(stokes.∇V)
        # ------------------------------
    end
    return nothing
end

ar = 1 # aspect ratio
n = 1
nx = 32 * n
ny = 32 * n
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

sinking_block2D(igg; ar = ar, nx = nx, ny = ny);
