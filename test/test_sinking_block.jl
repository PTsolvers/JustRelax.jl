push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end


using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC, JustPIC._2D
const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

using GeoParams

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)

    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        @inbounds if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            depth = abs(y[j])
            dTdZ = (2047 - 2017) / 50.0e3
            offset = 2017
            T[i + 1, j] = (depth - 585.0e3) * dTdZ + offset
        end
        return nothing
    end
    ni = length.(xvi)
    @parallel (@idx ni) _rectangular_perturbation!(T, xc, yc, r, xvi...)

    return nothing
end

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
function Sinking_Block2D()
    ar = 1
    n = 32
    nx = n
    ny = n
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain ------------------------------------
    ly = 500.0e3
    lx = ly * ar
    origin = 0.0, -ly                         # origin coordinates
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

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 40, 12
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    # temperature
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    # Rectangular density anomaly
    xc_anomaly = 250.0e3   # origin of thermal anomaly
    yc_anomaly = -(ly - 400.0e3) # origin of thermal anomaly
    r_anomaly = 50.0e3   # radius of perturbation
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, xc_anomaly, abs(yc_anomaly), r_anomaly)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-5, CFL = 0.95 / √2.1)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, (T = @ones(ni...), P = stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # ----------------------------------------------------

    # Viscosity
    args = (; T = @ones(ni...), P = stokes.P, dt = Inf)
    η_cutoff = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    # ----------------------------------------------------

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

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
        dt,
        igg;
        kwargs = (
            iterMax = 150.0e3,
            nout = 1.0e3,
            viscosity_cutoff = η_cutoff,
            verbose = false,
        )
    )
    dt = compute_dt(stokes, di, igg)
    # ------------------------------

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    velocity = @. √(Vx_v^2 + Vy_v^2)

    finalize_global_grid(; finalize_MPI = true)

    return iters, velocity
end

@testset "Sinking_Block2D" begin
    @suppress begin
        iters, velocity = Sinking_Block2D()
        @test passed = iters.err_evo1[end] < 1.0e-5
        @test maximum(velocity) ≈ 4.841885609356093e-10 atol = 1.0e-6
    end
end
