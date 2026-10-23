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

@parallel_indices (i, j) function init_block!(phases, xc, yc)
    inside = abs(xc[i] - 0.5) ≤ 0.2 && abs(yc[j] - 0.5) ≤ 0.15
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

# Objective J and the adjoint dJ/dρ of the block density, for a compressible sinking block: a
# finite bulk modulus makes the pressure unique, so a pressure objective is well posed.
function sinking_block_objective(ρ_block, observation, igg; variational, adjoint)
    ni = (24, 24)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    elasticity = ConstantElasticity(; G = 1.0e6, Kb = 2.0)
    material(phase, ρ, η) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ),
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((LinearViscous(; η), elasticity)),
        Gravity = ConstantGravity(; g = 1.0),
    )
    rheology = (material(1, 1.0, 1.0), material(2, ρ_block, 10.0))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_block!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_block!(phase_ratios.vertex, grid.xvi...)

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    dt = 1.0
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)

    gradients = material_controls(backend, ni, (:ρ,); nphases = length(rheology))
    solver_kwargs = (;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax = 2.0e5,
        nout = 50, linear_viscosity = true, adjoint, observation, gradients,
    )
    if variational
        ϕ = RockRatio(backend, ni)
        foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-9)
        solve_VariationalDYREL!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid,
            dt, igg; solver_kwargs...,
        )
    else
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-9)
        solve_DYREL!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid,
            dt, igg; solver_kwargs...,
        )
    end

    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    field = observation.field === :P ? stokes.P : getproperty(stokes.V, observation.field)
    J = sum(Array(field)[first(mask.i):last(mask.i), first(mask.j):last(mask.j)])
    return J, sum(Array(gradients.ρ.center)[2, :, :])
end

@testset "Adjoint gradient vs finite difference" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(24, 24, 1; init_MPI = init_mpi)...)
    ρ_block, h = 1.2, 1.0e-4
    for variational in (false, true), field in (:Vy, :P)
        observation = (; field, center = (0.5, 0.75), half_width = (0.1, 0.1))
        objective(ρ; adjoint = false) = @suppress sinking_block_objective(
            ρ, observation, igg; variational, adjoint
        )
        _, adjoint_gradient = objective(ρ_block; adjoint = true)
        J₊, = objective(ρ_block + h)
        J₋, = objective(ρ_block - h)
        @testset "$(variational ? "variational" : "standard") $field" begin
            @test adjoint_gradient ≈ (J₊ - J₋) / 2h rtol = 1.0e-6
        end
    end
    finalize_global_grid(; finalize_MPI = false)
end
