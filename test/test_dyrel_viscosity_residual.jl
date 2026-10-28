push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

const TA = PTArray(backend)

# Internal kernels must come from the module initialized for the selected backend.
const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

@parallel_indices (i, j) function init_inclusion!(phases, xc, yc)
    inside = (xc[i] - 0.5)^2 + (yc[j] - 0.5)^2 ≤ 0.1^2
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

# Pure shear of a weak shear-thinning inclusion (GeoParams' `PowerlawViscous` with n < 1),
# started from a homogeneous shear with uniform η. The stress of that start balances exactly, so
# the momentum and continuity residuals vanish although η is far from the viscosity its stress
# implies; the solve must not stop there. Returns the converged η and the viscosity h(τ(η)).
function shear_thinning_inclusion(igg; variational)
    ni = (16, 16)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    dt = 0.25
    cutoff = (1.0e-2, 1.0e2)
    elasticity = ConstantElasticity(; G = 1.0, Kb = 1.0e6)
    material(phase, η0) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ = 0.0),
        Gravity = ConstantGravity(; g = 0.0),
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology(
            (PowerlawViscous(; η0, n = 1 / 3, ε0 = 1.0), elasticity)
        ),
    )
    rheology = (material(1, 1.0), material(2, 0.1))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_inclusion!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_inclusion!(phase_ratios.vertex, grid.xvi...)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt)
    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    xv, yv = Array.(grid.xvi)
    stokes.V.Vx .= TA([x - 0.5 for x in xv, _ in 1:(ni[2] + 2)])
    stokes.V.Vy .= TA([-(y - 0.5) for _ in 1:(ni[1] + 2), y in yv])
    flow_bcs!(stokes, flow_bcs)

    solver_kwargs = (;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax = 4.0e5,
        nout = 100, rel_drop = 1.0e-2, linear_viscosity = false, viscosity_relaxation = 0.5,
        viscosity_cutoff = cutoff,
    )
    if variational
        ϕ = RockRatio(backend, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-8)
        solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            solver_kwargs...,
        )
    else
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)
        solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            solver_kwargs...,
        )
    end

    # The solve moves the stress into the history; restore the history it solved with.
    for τ_o in (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy)
        fill!(τ_o, 0.0)
    end
    # stress from η and h(τ) from that stress, unrelaxed
    η = Array(stokes.viscosity.η)
    JR2K.compute_stress_viscosity_DYREL!(
        stokes, @zeros(ni...), @zeros(ni...), rheology, phase_ratios, 1.0, dt, 1.0, args, cutoff, false,
    )
    return η, Array(stokes.viscosity.η)
end

@testset "DYREL converges the nonlinear viscosity" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(16, 16, 1; init_MPI = init_mpi)...)
    @testset "$(variational ? "variational" : "standard") solver" for variational in (false, true)
        η, h = @suppress shear_thinning_inclusion(igg; variational)
        @test maximum(abs.(h .- η) ./ η) < 1.0e-6
    end
    finalize_global_grid(; finalize_MPI = false)
end
