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
const long_tests = get(ENV, "JULIA_JUSTRELAX_LONG_TESTS", "0") == "1"

# End-to-end checks of the material gradients against central differences of the forward
# solve, one adjoint solve per case. Each case exercises a different adjoint path.

central_difference(f, x, h) = (f(x + h) - f(x - h)) / 2h

perturbed(p, name, value) = merge(p, NamedTuple{(name,)}((value,)))

# Check all gradients of a case with one central difference, along a direction through every
# parameter at once: each is scaled by its magnitude and gets its own weight and sign, so an
# error in any one of them changes the sum. The tolerance is relative to the sum of the absolute
# contributions, which alternate in sign. With `JULIA_JUSTRELAX_LONG_TESTS=1`, every parameter
# is also checked on its own, to locate a failure.
function check_gradients(objective, parameters, gradient, names; h, rtol = 1.0e-6)
    weights = map(
        (name, k) -> (-1)^k * (1 + k / 4) * max(abs(parameters[name]), 1.0),
        names, Tuple(eachindex(names)),
    )
    along(t) = merge(
        parameters, NamedTuple{names}(map((name, w) -> parameters[name] + t * w, names, weights))
    )
    terms = map((name, w) -> gradient[name] * w, names, weights)
    fd = central_difference(t -> objective(along(t)), 0.0, h)
    @test abs(sum(terms) - fd) ≤ rtol * sum(abs, terms)
    if long_tests
        @testset "$name" for name in names
            step = h * max(abs(parameters[name]), 1.0)
            fd_name = central_difference(parameters[name], step) do value
                objective(perturbed(parameters, name, value))
            end
            @test gradient[name] ≈ fd_name rtol = rtol
        end
    end
    return nothing
end

@parallel_indices (i, j) function init_inclusion!(phases, xc, yc)
    inside = (xc[i] - 0.5)^2 + (yc[j] - 0.5)^2 ≤ 0.15^2
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

const shear_parameters = (;
    η0 = 1.0, n = 2.0, G = 1.0, Kb = 5.0, C = 0.25, ϕ = 30.0, Ψ = 10.0, α = 0.02, pT = -0.05,
)

# One viscoelastic-plastic step of pure shear around a weak inclusion. The matrix (phase 1)
# carries the parameters `p`; a power-law viscosity starts from η = 1, since it is infinite
# at the initial zero stress. Thermal expansion enters the continuity residual through ΔT.
# Returns J for a weighted Vx objective and the adjoint gradients of the matrix parameters.
function plastic_shear(igg, p, names; plasticity, adjoint = false, ϵ = 1.0e-10, cells = 16)
    ni = (cells, cells)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    xc, yc = Array.(grid.xci)
    xv, yv = Array.(grid.xvi)
    dt = 0.25
    function material(phase, q)
        elasticity = ConstantElasticity(; G = q.G, Kb = q.Kb)
        return SetMaterialParams(;
            Phase = phase,
            Density = PT_Density(; ρ0 = 1.0, α = q.α, β = 0.0, T0 = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology(
                (PowerlawViscous(; η0 = q.η0, n = q.n, ε0 = 1.0), elasticity, plasticity(q))
            ),
            Elasticity = elasticity,
        )
    end
    rheology = (material(1, p), material(2, merge(shear_parameters, (; G = 0.5, C = 0.4))))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_inclusion!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_inclusion!(phase_ratios.vertex, grid.xvi...)

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    xg, yg = [xc[1]; xc; xc[end]], [yc[1]; yc; yc[end]]
    ΔT = TA([0.1 * sinpi(x) * sinpi(y) for x in xg, y in yg])
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt, ΔT)
    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    Vx = [x for x in xv, _ in 1:(ni[2] + 2)]
    Vy = [-y for _ in 1:(ni[1] + 2), y in yv]
    Vx[2:(end - 1), 2:(end - 1)] .= 0.0
    Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    stokes.V.Vx .= TA(Vx)
    stokes.V.Vy .= TA(Vy)
    flow_bcs!(stokes, flow_bcs)

    # Vx lives on x-faces, with a ghost row below and above the domain
    yVx = [yv[1]; (yv[1:(end - 1)] .+ yv[2:end]) ./ 2; yv[end]]
    weights = [exp(-((x - 0.7)^2 + (y - 0.6)^2) / 0.02) for x in xv, y in yVx]
    observation = (; field = :Vx, weights = TA(weights))
    gradients = material_controls(backend, ni, names; nphases = length(rheology))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ)
    solve_DYREL!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax = 4.0e5,
        nout = 50, rel_drop = 0.5, viscosity_relaxation = 0.5, linear_viscosity = false,
        viscosity_cutoff = (1.0e-2, 1.0e2), adjoint, observation, gradients,
    )
    η = Array(stokes.viscosity.η)
    return (;
        J = sum(weights .* Array(stokes.V.Vx)),
        gradient = map(name -> sum(Array(gradients[name].center)[1, :, :]), NamedTuple{names}(names)),
        yielded = count(>(0), Array(stokes.λ)),
        clamped = count(x -> x ≤ 1.0e-2 || x ≥ 1.0e2, η),
    )
end

# phase 1 matrix, 2 block, 3 sticky air above y = 0.8
@parallel_indices (i, j) function init_air_block!(phases, xc, yc)
    air = yc[j] > 0.8
    block = !air && abs(xc[i] - 0.4) ≤ 0.15 && abs(yc[j] - 0.45) ≤ 0.12
    @index phases[1, i, j] = (air || block) ? 0.0 : 1.0
    @index phases[2, i, j] = block ? 1.0 : 0.0
    @index phases[3, i, j] = air ? 1.0 : 0.0
    return nothing
end

const block_parameters = (; ρ_block = 1.2, η_block = 10.0, ρ_matrix = 1.0, η_matrix = 1.0)

# A block sinking below sticky air, through the variational solver with free-surface
# stabilization and a periodic x axis. Returns J and the density and viscosity gradients.
function sinking_block_air(igg, p, observation; adjoint = false)
    ni = (16, 16)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    dt = 1.0
    elasticity = ConstantElasticity(; G = 1.0e6, Kb = 2.0)
    material(phase, ρ, η) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ),
        Gravity = ConstantGravity(; g = 1.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η), elasticity)),
        Elasticity = elasticity,
    )
    rheology = (
        material(1, p.ρ_matrix, p.η_matrix), material(2, p.ρ_block, p.η_block),
        material(3, 0.0, 1.0e-2),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_air_block!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_air_block!(phase_ratios.vertex, grid.xvi...)
    air_phase = 3
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, air_phase)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
        periodic = (left = true, right = true, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt)
    compute_ρg!(ρg, phase_ratios, rheology, args; air_phase)
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, (-Inf, Inf); air_phase)
    flow_bcs!(stokes, flow_bcs)
    gradients = material_controls(backend, ni, (:ρ, :η); nphases = length(rheology))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-9)
    solve_VariationalDYREL!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
        air_phase, free_surface = true, verbose_PH = false, verbose_DR = false,
        iterMax = 2.0e5, total_iterMax = 4.0e5, nout = 50, linear_viscosity = true,
        adjoint, observation, gradients,
    )
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    field = observation.field === :P ? stokes.P : getproperty(stokes.V, observation.field)
    J = sum(Array(field)[first(mask.i):last(mask.i), first(mask.j):last(mask.j)])
    phase_gradient(G, phase) = sum(Array(G.center)[phase, :, :])
    gradient = (;
        ρ_block = phase_gradient(gradients.ρ, 2), η_block = phase_gradient(gradients.η, 2),
        ρ_matrix = phase_gradient(gradients.ρ, 1), η_matrix = phase_gradient(gradients.η, 1),
    )
    return (; J, gradient)
end

@parallel_indices (i, j) function init_dense_block!(phases, xc, yc)
    block = abs(xc[i] - 0.4) ≤ 0.15 && abs(yc[j] - 0.55) ≤ 0.12
    @index phases[1, i, j] = block ? 0.0 : 1.0
    @index phases[2, i, j] = block ? 1.0 : 0.0
    return nothing
end

const variational_parameters = (; η0 = 1.0, n = 2.0, C = 0.02, ϕ = 20.0, ρ0 = 1.0, β = 0.05, g = 1.0)

# A dense block in background shear, through the variational solver with no-slip at the bottom,
# a pressure-dependent density and either a power-law viscosity or (linear viscosity and)
# Drucker–Prager plasticity with Ψ = 20°, associated at ϕ = 20°. The matrix (phase 1) carries
# `p`; the block has fixed parameters, and gravity is taken from phase 1. Rock fills the
# domain: sticky air with a power law, or strong non-associated plasticity, does not converge
# in the forward or the adjoint solve.
function variational_shear(igg, p, names; plastic, adjoint = false)
    ni = (16, 16)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    xv, yv = Array.(grid.xvi)
    dt = 0.5
    elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
    function material(phase; η0, n, C, ϕ, ρ0, β, g)
        viscous = plastic ? LinearViscous(; η = η0) : PowerlawViscous(; η0, n, ε0 = 1.0)
        plasticity = DruckerPrager_regularised(; C, ϕ, Ψ = 20.0, η_vp = 1.0e-2)
        return SetMaterialParams(;
            Phase = phase,
            Density = PT_Density(; ρ0, β, α = 0.0, T0 = 0.0),
            Gravity = ConstantGravity(; g),
            CompositeRheology = CompositeRheology(
                plastic ? (viscous, elasticity, plasticity) : (viscous, elasticity)
            ),
            Elasticity = elasticity,
        )
    end
    rheology = (
        material(1; p...),
        material(2; η0 = 1.0, n = 2.0, C = 1.0, ϕ = 20.0, ρ0 = 1.5, β = 0.02, g = p.g),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_dense_block!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_dense_block!(phase_ratios.vertex, grid.xvi...)
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, 0)

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = false),
        no_slip = (left = false, right = false, top = false, bot = true),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    compute_ρg!(ρg, phase_ratios, rheology, args)
    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    # a stress history, so that the first trial stress of the τII power law is not zero
    foreach(
        (A, value) -> fill!(A, value),
        (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy),
        (0.05, -0.05, 0.02, 0.05, -0.05, 0.02),
    )
    Vx = [x - 0.5 for x in xv, _ in 1:(ni[2] + 2)]
    Vy = [-y for _ in 1:(ni[1] + 2), y in yv]
    Vx[2:(end - 1), 2:(end - 1)] .= 0.0
    Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    stokes.V.Vx .= TA(Vx)
    stokes.V.Vy .= TA(Vy)
    flow_bcs!(stokes, flow_bcs)

    observation = (; field = :Vy, center = (0.6, 0.55), half_width = (0.1, 0.1))
    gradients = material_controls(backend, ni, names; nphases = length(rheology))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-10)
    solve_VariationalDYREL!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
        free_surface = true, verbose_PH = false, verbose_DR = false, iterMax = 2.0e5,
        total_iterMax = 4.0e5, nout = 50, rel_drop = 0.5, linear_viscosity = plastic,
        viscosity_relaxation = 0.5, viscosity_cutoff = (1.0e-2, 1.0e2),
        adjoint, observation, gradients,
    )
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    η = Array(stokes.viscosity.η)
    return (;
        J = sum(Array(stokes.V.Vy)[first(mask.i):last(mask.i), first(mask.j):last(mask.j)]),
        gradient = map(name -> sum(Array(gradients[name].center)[1, :, :]), NamedTuple{names}(names)),
        yielded = count(>(0), Array(stokes.λ)),
        clamped = count(x -> x ≤ 1.0e-2 || x ≥ 1.0e2, η),
    )
end

@testset "Adjoint gradients through the physics paths" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    # `Geometry` takes its spacing from the global grid, so each grid size needs its own.
    igg = IGG(init_global_grid(16, 16, 1; init_MPI = init_mpi)...)

    # Nonlinear viscosity (implicit viscosity feedback), elasticity, Drucker–Prager with the
    # sin/cos chain rule for ϕ and Ψ, compressibility, thermal expansion, weighted objective.
    @testset "power law + Drucker–Prager" begin
        names = (:η0, :n, :G, :Kb, :C, :ϕ, :Ψ, :α)
        drucker_prager(q) = DruckerPrager_regularised(; C = q.C, ϕ = q.ϕ, Ψ = q.Ψ, η_vp = 1.0e-2)
        solution = @suppress plastic_shear(
            igg, shear_parameters, names; plasticity = drucker_prager, adjoint = true
        )
        @test solution.yielded > 0
        @test iszero(solution.clamped)
        check_gradients(shear_parameters, solution.gradient, names; h = 1.0e-5) do p
            @suppress plastic_shear(igg, p, names; plasticity = drucker_prager).J
        end
    end

    # Variational solver: sticky air, free-surface stabilization, periodic x axis.
    @testset "variational, sticky air, $field" for field in (:Vy, :P)
        observation = (; field, center = (0.6, 0.6), half_width = (0.1, 0.1))
        solution = @suppress sinking_block_air(igg, block_parameters, observation; adjoint = true)
        check_gradients(block_parameters, solution.gradient, keys(block_parameters); h = 1.0e-5) do p
            @suppress sinking_block_air(igg, p, observation).J
        end
    end

    # Variational solver with nonlinear viscosity or plasticity, no-slip at the bottom, and the
    # density compressibility β and gravity g, which reach the residual through ρg(P).
    @testset "variational, $(plastic ? "Drucker–Prager" : "power law"), no-slip bottom" for plastic in (false, true)
        names = plastic ? (:C, :ϕ, :ρ0, :β, :g) : (:η0, :n, :ρ0, :β, :g)
        solution = @suppress variational_shear(igg, variational_parameters, names; plastic, adjoint = true)
        @test iszero(solution.clamped)
        plastic && @test solution.yielded > 0
        # the nonlinear solves converge to ~1e-10, so a smaller step is dominated by noise
        check_gradients(variational_parameters, solution.gradient, names; h = 1.0e-4) do p
            @suppress variational_shear(igg, p, names; plastic).J
        end
    end
    finalize_global_grid(; finalize_MPI = false)

    # The cap return map: an inactive local Newton solve and one differentiated implicit step.
    # Its forward solves are the slowest, so it runs on a coarser grid.
    @testset "power law + Drucker–Prager cap" begin
        cells = 12
        igg = IGG(init_global_grid(cells, cells, 1; init_MPI = false)...)
        names = (:C, :pT)
        parameters = merge(shear_parameters, (; Ψ = 0.0))
        cap(q) = DruckerPragerCap(; C = q.C, ϕ = q.ϕ, Ψ = q.Ψ, η_vp = 1.0e-2, pT = q.pT)
        solution = @suppress plastic_shear(
            igg, parameters, names; plasticity = cap, adjoint = true, cells
        )
        @test solution.yielded > 0
        check_gradients(parameters, solution.gradient, names; h = 1.0e-5) do p
            @suppress plastic_shear(igg, p, names; plasticity = cap, cells).J
        end
        finalize_global_grid(; finalize_MPI = false)
    end
end
