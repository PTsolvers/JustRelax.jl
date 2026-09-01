const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg; Pkg.activate("miniapps")

const backend_JR = @static if isCUDA
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
const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams, CairoMakie

# x-length of the domain
const λ = 0.9142

# HELPER FUNCTIONS ---------------------------------------------------------------
# Initialize phases on the particles
function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = @index py[ip, i, j]

            # plume - rectangular
            if y > 0.2 + 0.02 * cos(π * x / λ)
                @index phases[ip, i, j] = 2.0
            else
                @index phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
# END OF HELPER FUNCTIONS --------------------------------------------------------

# MAIN SCRIPT --------------------------------------------------------------------
function main2D(
        igg; ny = 64, nx = 64, figdir = "model_figs", data_dir = "model_data",
        plot_progress = true, snapshot_fracs = ()
    )

    # Physical domain ------------------------------------
    ly = 1            # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = 1.0e-10

    # Physical properties using GeoParams ----------------
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1),
            Gravity = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e0),)),

        ),
        # High density phase
        SetMaterialParams(;
            Density = ConstantDensity(; ρ = 2),
            Gravity = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e0),)),
        ),
    )

    # Initialize particles -------------------------------
    nxcell, max_p, min_p = 20, 30, 10
    particles = init_particles(
        backend, nxcell, max_p, min_p, grid.xi_vel...
    )
    # temperature
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)

    # Rheology
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = false, bot = false),
        no_slip = (left = false, right = false, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    !isdir(data_dir) && mkpath(data_dir)
    # ----------------------------------------------------

    # Buffer arrays to compute velocity rms
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    tmax = 2.0e3
    Urms = Float64[]
    trms = Float64[]
    sizehint!(Urms, 100000)
    sizehint!(trms, 100000)

    # snapshot times (density field checkpoints), consumed in order as t crosses each target
    snapshot_targets = sort(collect(snapshot_fracs) .* tmax)

    # DYREL solver state (rebuilt from the current rheology inside solve_DYREL!)
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)

    while t < tmax

        # Update buoyancy
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        # ------------------------------

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
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 50.0e3,
                nout = 100,
                # verbose_PH = true,
                # verbose_DR = false,
                # iterMax = 50.0e3,
                # nout = 50,
                # rel_drop = 0.1,
                # λ_relaxation_PH = 1,
                # λ_relaxation_DR = 1,
                linear_viscosity = true,
                # viscosity_relaxation = 1.0e-2,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        dt = compute_dt(stokes, di) * 0.8
        # ------------------------------

        # Compute U rms ---------------
        Urms_it = let
            velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sum(Vx_v .^ 2) * prod(di) |> sqrt
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # ------------------------------

        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        # # advect particles in memory
        move_particles!(particles, particle_args)
        # inject && break
        inject_particles_phase!(particles, pPhases, (), ())
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)

        @show it += 1
        t += dt

        # Snapshot checkpoints (density field), taken as t crosses each requested target time
        if !isempty(snapshot_targets) && t >= snapshot_targets[1]
            target = popfirst!(snapshot_targets)
            fname = joinpath(data_dir, "snapshot_$(round(Int, target))_$(nx)x$(ny).jld2")
            checkpointing_jld2(
                data_dir, stokes, nothing, t, it, fname;
                ρg = Array(ρg[2]), xci = Array.(xci), xvi = Array.(xvi)
            )
        end

        # Plotting ---------------------
        if plot_progress && (it == 1 || rem(it, 25) == 0 || t >= tmax)
            fig = Figure(size = (1000, 1000), font = "TeX Gyre Heros Makie")
            ax1 = Axis(
                fig[1:2, 1], aspect = 1 / λ, title = "VanKeken ",
                titlesize = 20,
                yticklabelsize = 12,
                xticklabelsize = 12,
                xlabelsize = 12,
                ylabelsize = 12
            )
            h = heatmap!(ax1, xvi[1], xvi[2], Array(ρg[2]), colormap = :lapaz)
            Colorbar(fig[1:2, 2], h; height = Relative(1.0), label = "Density", labelsize = 20, ticklabelsize = 12)
            ax2 = Axis(
                fig[3, 1], aspect = 2.25, xlabel = L"Time", ylabel = L"V_{RMS}",
                titlesize = 20,
                yticklabelsize = 12,
                xticklabelsize = 12,
                xlabelsize = 12,
                ylabelsize = 12
            )
            ylims!(ax2, 0, 0.005)
            lines!(ax2, trms, Urms, color = :black)

            save(joinpath(figdir, "$(it).png"), fig)
            fig
        end

    end

    # Final checkpoint: Urms/trms time series (every resolution) plus the last density field
    checkpointing_jld2(
        data_dir, stokes, nothing, t, it, joinpath(data_dir, "final_$(nx)x$(ny).jld2");
        Urms = Urms, trms = trms, ρg = Array(ρg[2]), xci = Array.(xci), xvi = Array.(xvi)
    )

    return nothing
end

figdir = "VanKeken_DYREL"
n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
nx = n
ny = n
save_snapshots = "snapshots" in ARGS
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main2D(
    igg; figdir = figdir, nx = nx, ny = ny,
    data_dir = "VanKeken_DYREL_data", plot_progress = isempty(ARGS),
    snapshot_fracs = save_snapshots ? (0.1, 0.5, 0.9) : ()
);
