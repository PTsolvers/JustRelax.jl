push!(LOAD_PATH, "..")
using Test, Suppressor

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

using Printf, LinearAlgebra, GeoParams, CellArrays
using JustRelax, JustRelax.DataIO
import JustRelax.@cell
using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

# x-length of the domain
const λ = 0.9142

# HELPER FUNCTIONS ---------------------------------------------------------------
import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

# Initial pressure guess
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_j(z)) * <(@all_j(z), 0.0)
    return nothing
end

# Initialize phases on the particles
function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = JustRelax.@cell py[ip, i, j]

            # plume - rectangular
            if y > 0.2 + 0.02 * cos(π * x / λ)
                JustRelax.@cell phases[ip, i, j] = 2.0
            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
# END OF HELPER FUNCTIONS --------------------------------------------------------


# MAIN SCRIPT --------------------------------------------------------------------
function VanKeken2D(ny=32, nx=32)

    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg    = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain ------------------------------------
    ly           = 1            # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt           = 1

    # Physical properties using GeoParams ----------------
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 1),
            Gravity           = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1e0),)),

        ),
        # High density phase
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ = 2),
            Gravity           = ConstantGravity(; g = 1),
            CompositeRheology = CompositeRheology((LinearViscous(;η = 1e0),)),
        ),
    )

    # Initialize particles -------------------------------
    nxcell, max_p, min_p = 40, 40, 15
    particles            = init_particles(
        backend, nxcell, max_p, min_p, xvi..., di..., nx, ny
    )
    # velocity grids
    grid_vx, grid_vy     = velocity_grids(xci, xvi, di)
    # temperature
    pPhases,             = init_cell_arrays(particles, Val(1))
    particle_args        = (pPhases, )
    init_phases!(pPhases, particles)
    phase_ratios         = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes               = StokesArrays(ni, ViscoElastic)
    pt_stokes            = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 0.9 / √2.1)

    # Buoyancy forces
    ρg                   = @zeros(ni...), @zeros(ni...)
    args                 = (; T = @zeros(ni...), P = stokes.P, dt = Inf)
    @parallel (JustRelax.@idx ni) JustRelax.compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
    @parallel init_P!(stokes.P, ρg[2], xci[2])

    # Rheology
    η                    = @ones(ni...)
    η_vep                = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs             = FlowBoundaryConditions(;
        free_slip = (left =  true, right =  true, top = false, bot = false),
        no_slip   = (left = false, right = false, top =  true, bot =  true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    # Buffer arrays to compute velocity rms
    Vx_v  = @zeros(ni.+1...)
    Vy_v  = @zeros(ni.+1...)

    # Time loop
    t, it = 0.0, 0
    tmax  = 0.5e3
    nt    = 500
    Urms  = Float64[]
    trms  = Float64[]
    sizehint!(Urms, 100000)
    sizehint!(trms, 100000)
    local iters, Urms

    while it < nt

        # Update buoyancy
        @parallel (JustRelax.@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax          = 10e3,
            nout             = 50,
            viscosity_cutoff = (-Inf, Inf)
        )
        dt = compute_dt(stokes, di) / 10
        # ------------------------------

        # Compute U rms ---------------
        Urms_it = let
            JustRelax.velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy; ghost_nodes=true)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sum(Vx_v.^2) * prod(di) |> sqrt
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # ------------------------------

        # advect particles in space
        advection_RK!(particles, @velocity(stokes), grid_vx, grid_vy, dt, 2 / 3)
        # # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        # inject && break
        inject && inject_particles_phase!(particles, pPhases, (), (), xvi)
        # update phase ratios
        @parallel (@idx ni) phase_ratios_center(phase_ratios.center, pPhases)

        @show it += 1
        t        += dt
    end

    return iters, Urms
end

@testset "VanKeken" begin
    @suppress begin
        iters, Urms = VanKeken2D()
        @test passed = iters.err_evo1[end] < 1e-4
        @test all(<(1e-2), Urms) 
    end
end
