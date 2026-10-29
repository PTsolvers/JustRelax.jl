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

@parallel_indices (i, j) function init_inclusion!(phases, xc, yc)
    inside = (xc[i] - 0.5)^2 + (yc[j] - 0.5)^2 ≤ 0.12^2
    @index phases[1, i, j] = inside ? 0.0 : 1.0
    @index phases[2, i, j] = inside ? 1.0 : 0.0
    return nothing
end

observed_sum(field, mask) =
    sum(Array(field)[first(mask.i):last(mask.i), first(mask.j):last(mask.j)])

# Compressible sinking block: a finite bulk modulus makes the pressure unique, so a pressure
# objective is well posed. Returns J, the adjoint dJ/d`parameter` of the block (phase 2), and the
# viscosity left in `stokes` after the solve. `solves` time steps run on the same arrays; before
# each, λ is kept (`λ_start = :previous`), scaled by 1000 (`:scaled`), zeroed (`:zero`) or
# poisoned with NaN (`:nan`).
# With an `activation_volume`, both phases creep with η ∝ exp(V P / R T) instead. With a
# `compressibility` β the density is ρ (1 + β P); `update_material` goes to the standard solver.
function sinking_block(
        igg; observation, variational = false, adjoint = false, parameter = :ρ,
        ρ_block = 1.2, η_block = 10.0, η_multiplier = nothing, total_iterMax = 2.0e5,
        solves = 1, λ_start = :previous, adjoint_ϵ = nothing, activation_volume = nothing,
        compressibility = nothing, update_material = true, adjoint_kwargs = (;),
    )
    ni = (24, 24)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    elasticity = ConstantElasticity(; G = 1.0e6, Kb = 2.0)
    linear_viscosity = isnothing(activation_volume)
    viscous(η) = linear_viscosity ? LinearViscous(; η) :
        DislocationCreep(; n = 1.0, r = 0.0, A = 0.5 / η, E = 0.0, V = activation_volume)
    material(phase, ρ, η) = SetMaterialParams(;
        Phase = phase,
        Density = isnothing(compressibility) ? ConstantDensity(; ρ) :
            PT_Density(; ρ0 = ρ, β = compressibility, α = 0.0, T0 = 0.0),
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((viscous(η), elasticity)),
        Gravity = ConstantGravity(; g = 1.0),
    )
    rheology = (material(1, 1.0, 1.0), material(2, ρ_block, η_block))
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

    gradients = material_controls(backend, ni, (parameter,); nphases = length(rheology))
    solver_kwargs = (;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax,
        nout = 50, linear_viscosity, adjoint, observation, gradients, adjoint_ϵ,
        adjoint_kwargs,
    )
    if variational
        ϕ = RockRatio(backend, ni)
        foreach(A -> fill!(A, 1.0), (ϕ.center, ϕ.vertex, ϕ.Vx, ϕ.Vy))
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-9)
        for _ in 1:solves
            set_adjoint_start!(stokes_ad, λ_start)
            solve_VariationalDYREL!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid,
                dt, igg; solver_kwargs...,
            )
        end
    else
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-9)
        for _ in 1:solves
            set_adjoint_start!(stokes_ad, λ_start)
            solve_DYREL!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid,
                dt, igg; solver_kwargs..., η_multiplier, update_material,
            )
        end
    end

    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    field = observation.field === :P ? stokes.P : getproperty(stokes.V, observation.field)
    return (;
        J = observed_sum(field, mask),
        gradient = sum(Array(gradients[parameter].center)[2, :, :]),
        η = Array(stokes.viscosity.η),
        dyrel,
        stokes_ad,
    )
end

# One viscoelastic-plastic (Drucker–Prager) step of pure shear around a weak inclusion, with the
# plastic multiplier relaxed by `λ_relaxation`. Returns J and the adjoint dJ/dC of the matrix.
function plastic_shear(
        igg; observation, C_matrix = 0.25, λ_relaxation = 1, adjoint = false, cells = 16,
    )
    ni = (cells, cells)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    xv, yv = Array.(grid.xvi)
    dt = 0.25
    function material(phase, G, C)
        elasticity = ConstantElasticity(; G, Kb = 5.0)
        plasticity = DruckerPrager_regularised(; C, ϕ = 30.0, η_vp = 1.0e-2, Ψ = 0.0)
        return SetMaterialParams(;
            Phase = phase,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity, plasticity)),
            Elasticity = elasticity,
        )
    end
    rheology = (material(1, 1.0, C_matrix), material(2, 0.5, 0.3))
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    @parallel (@idx ni) init_inclusion!(phase_ratios.center, grid.xci...)
    @parallel (@idx ni .+ 1) init_inclusion!(phase_ratios.vertex, grid.xvi...)

    stokes = StokesArrays(backend, ni)
    stokes_ad = AdjointStokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    Vx = [x for x in xv, _ in 1:(ni[2] + 2)]
    Vy = [-y for _ in 1:(ni[1] + 2), y in yv]
    Vx[2:(end - 1), 2:(end - 1)] .= 0.0
    Vy[2:(end - 1), 2:(end - 1)] .= 0.0
    stokes.V.Vx .= PTArray(backend)(Vx)
    stokes.V.Vy .= PTArray(backend)(Vy)
    flow_bcs!(stokes, flow_bcs)

    gradients = material_controls(backend, ni, (:C,); nphases = length(rheology))
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-10)
    solve_DYREL!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        verbose_PH = false, verbose_DR = false, iterMax = 2.0e5, total_iterMax = 4.0e5,
        nout = 50, rel_drop = 0.5, λ_relaxation_PH = λ_relaxation,
        λ_relaxation_DR = λ_relaxation, viscosity_relaxation = 1, linear_viscosity = true,
        adjoint, observation, gradients,
    )
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    return (;
        J = observed_sum(stokes.V.Vx, mask),
        gradient = sum(Array(gradients.C.center)[1, :, :]),
        yielded = count(>(0), Array(stokes.λ)),
    )
end

function set_adjoint_start!(adjoint, λ_start)
    λ = (adjoint.λV.Vx, adjoint.λV.Vy, adjoint.λP)
    λ_start === :scaled && foreach(A -> A .*= 1.0e3, λ)
    λ_start in (:zero, :nan) && foreach(A -> fill!(A, λ_start === :zero ? 0.0 : NaN), λ)
    return nothing
end

central_difference(f, x, h) = (f(x + h) - f(x - h)) / 2h

@testset "Adjoint gradient vs finite difference" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(24, 24, 1; init_MPI = init_mpi)...)

    for variational in (false, true), field in (:Vy, :P)
        observation = (; field, center = (0.5, 0.75), half_width = (0.1, 0.1))
        adjoint_gradient = @suppress sinking_block(igg; observation, variational, adjoint = true).gradient
        fd = central_difference(1.2, 1.0e-4) do ρ_block
            @suppress sinking_block(igg; observation, variational, ρ_block).J
        end
        @testset "density, $(variational ? "variational" : "standard") $field" begin
            @test adjoint_gradient ≈ fd rtol = 1.0e-6
        end
    end

    @testset "viscosity" begin
        # J depends only weakly on the block viscosity (d ln J/d ln η ≈ 5e-3 here, less for a
        # stiffer block), so the difference quotient amplifies the forward residual; a larger
        # step keeps that below the tolerance.
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        adjoint_gradient = @suppress sinking_block(
            igg; observation, adjoint = true, parameter = :η, η_block = 0.5,
        ).gradient
        fd = central_difference(0.5, 1.0e-3) do η_block
            @suppress sinking_block(igg; observation, η_block).J
        end
        @test adjoint_gradient ≈ fd rtol = 1.0e-6
    end

    # With `update_material = false` the forward keeps the caller's ρg, so the adjoint must not
    # differentiate ρg(P) although the density depends on pressure. `sinking_block` computes η
    # from the rheology, so the η gradient holds; the solver warns that it assumes this.
    @testset "viscosity, fixed buoyancy" begin
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        case = (; observation, compressibility = 0.5, update_material = false)
        # `@test_logs` inside `@suppress`, which would otherwise swallow the warning
        adjoint_gradient = @suppress(
            @test_logs (:warn, r"update_material = false: the gradients of η") match_mode = :any sinking_block(
                igg; case..., adjoint = true, parameter = :η, η_block = 0.5,
            )
        ).gradient
        fd = central_difference(0.5, 1.0e-3) do η_block
            @suppress sinking_block(igg; case..., η_block).J
        end
        @test adjoint_gradient ≈ fd rtol = 1.0e-6
    end

    @testset "adjoint DYREL state, $(variational ? "variational" : "standard")" for variational in (false, true)
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        forward = @suppress sinking_block(igg; observation, variational)
        coupled = @suppress sinking_block(igg; observation, variational, adjoint = true)
        # the adjoint solve leaves the forward DYREL state as the forward solve left it
        for name in (:cVx, :cVy, :dτVx, :dτVy, :βVx, :αVx, :dVxdτ)
            @test Array(getfield(coupled.dyrel, name)) == Array(getfield(forward.dyrel, name))
        end
        state = coupled.stokes_ad.solver.dyrel
        @test state isa JustRelax.DYREL
        @test state.cVx !== coupled.dyrel.cVx
        # a later adjoint solve keeps the adjoint damping and takes the current forward operator
        fill!(state.cVx, 7.0)
        fill!(coupled.dyrel.Dx, 3.0)
        next = JustRelax2D.adjoint_dyrel!(coupled.stokes_ad, coupled.dyrel)
        @test next.cVx === state.cVx
        @test all(==(7.0), Array(next.cVx))
        @test all(==(3.0), Array(next.Dx))
        @test next.Dx !== coupled.dyrel.Dx
    end

    # The viscosity depends on the solved pressure only, so its pullback must reach λP.
    @testset "pressure-dependent viscosity, $(variational ? "variational" : "standard")" for variational in (false, true)
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        activation_volume = 8.3145 # V P / R T = P at T = 1
        solution = @suppress sinking_block(
            igg; observation, variational, adjoint = true, activation_volume,
        )
        fd = central_difference(1.2, 1.0e-4) do ρ_block
            @suppress(sinking_block(igg; observation, variational, ρ_block, activation_volume)).J
        end
        @test solution.gradient ≈ fd rtol = 1.0e-6
    end

    # a looser adjoint tolerance changes the gradient, not the forward solution
    @testset "adjoint tolerance, $(variational ? "variational" : "standard")" for variational in (false, true)
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        tight = @suppress sinking_block(igg; observation, variational, adjoint = true)
        loose = @suppress sinking_block(
            igg; observation, variational, adjoint = true, adjoint_ϵ = 1.0e-3,
        )
        @test loose.J == tight.J
        @test loose.gradient != tight.gradient
        @test loose.gradient ≈ tight.gradient rtol = 1.0e-2
        options = @suppress sinking_block(
            igg; observation, variational, adjoint = true, adjoint_kwargs = (; ϵ = 1.0e-3),
        )
        @test options.gradient == loose.gradient
    end

    @testset "adjoint solver options" begin
        forward = (; iterMax_DR = 10, nout = 100, rel_drop = 1.0e-2, verbose_PH = true)
        options(adjoint_kwargs; kwargs...) =
            JustRelax2D.adjoint_solver_options(forward, adjoint_kwargs, kwargs)
        @test options((;)) == forward
        @test options((; nout = 10, ϵ = 1.0e-6)) ==
            (; forward..., nout = 10, adjoint_ϵ = 1.0e-6)
        @test options((; iterMax = 5)).iterMax_DR == 5
        @test options((;); adjoint_rel_drop = 0.1).adjoint_rel_drop == 0.1
        @test_throws ArgumentError options((; viscosity_cutoff = (1, 2)))
        @test_throws ArgumentError options((; ϵ = 1.0e-6); adjoint_ϵ = 1.0e-6)
        @test options((; ϵ = 1.0e-6); adjoint_ϵ = nothing).adjoint_ϵ == 1.0e-6
        @test_throws ArgumentError options([:nout => 10])
    end

    # Starting from the previous time step's λ changes the iteration, not the result, and the λ
    # of a failed solve is replaced by zero.
    @testset "adjoint start, $(variational ? "variational" : "standard") $field" for variational in (false, true), field in (:Vy, :P)
        observation = (; field, center = (0.5, 0.75), half_width = (0.1, 0.1))
        warm, scaled, cold, poisoned = map((:previous, :scaled, :zero, :nan)) do λ_start
            @suppress sinking_block(
                igg; observation, variational, adjoint = true, solves = 2, λ_start,
            ).gradient
        end
        @test warm ≈ cold rtol = 1.0e-6
        @test scaled ≈ cold rtol = 1.0e-6
        @test poisoned ≈ cold rtol = 1.0e-6
    end

    # The initial λ is scaled to the multiple that minimizes the affine residual b + L λ.
    @testset "scaled adjoint start" begin
        stokes_ad = AdjointStokesArrays(backend, (6, 5))
        λ = (stokes_ad.λV.Vx, stokes_ad.λV.Vy)
        b = map(A -> fill!(similar(A), -1.0), λ)
        residuals = map(copy, b)
        evaluate!() = foreach((r, b, λ) -> r .= b .+ 2 .* λ, residuals, b, λ)
        masks = (nothing, nothing)
        foreach(A -> fill!(A, 0.0), (λ..., stokes_ad.λP))
        @test isnothing(JustRelax2D.scale_adjoint_start!(evaluate!, stokes_ad, residuals, masks))
        foreach(A -> fill!(A, 1.0e3), (λ..., stokes_ad.λP))
        α = JustRelax2D.scale_adjoint_start!(evaluate!, stokes_ad, residuals, masks)
        @test α ≈ 5.0e-4
        @test all(≈(0.5), Array(stokes_ad.λV.Vx))
        @test all(≈(0.5), Array(stokes_ad.λP))
    end

    # Sensitivities of an unconverged adjoint state are not returned.
    @testset "unconverged adjoint, $(variational ? "variational" : "standard")" for variational in (false, true)
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        @test_throws "adjoint did not converge" @suppress sinking_block(
            igg; observation, variational, adjoint = true, total_iterMax = 10,
        )
    end

    # The sensitivity pass must not replace the forward viscosity by the one of the rheology.
    @testset "forward viscosity kept" begin
        observation = (; field = :Vy, center = (0.5, 0.75), half_width = (0.1, 0.1))
        ni = (24, 24)
        η_multiplier = (; center = 2 .* @ones(ni...), vertex = 2 .* @ones((ni .+ 1)...))
        forward = @suppress sinking_block(igg; observation, η_multiplier)
        adjoint = @suppress sinking_block(
            igg; observation, η_multiplier, adjoint = true, parameter = :η,
        )
        @test adjoint.η == forward.η
    end

    finalize_global_grid(; finalize_MPI = false)

    # A relaxed plastic multiplier converges to the same state; its derivative must not depend
    # on the relaxation. `test_adjoint_gradients_physics.jl` covers Drucker–Prager without
    # relaxation, so that case runs only with `JULIA_JUSTRELAX_LONG_TESTS=1`.
    # `Geometry` takes its spacing from the global grid, so the coarser grid needs its own.
    cells = 16
    igg = IGG(init_global_grid(cells, cells, 1; init_MPI = false)...)
    long_tests = get(ENV, "JULIA_JUSTRELAX_LONG_TESTS", "0") == "1"
    @testset "plasticity, λ_relaxation = $λ_relaxation" for λ_relaxation in (long_tests ? (1, 0.5) : (0.5,))
        observation = (; field = :Vx, center = (0.7, 0.65), half_width = (0.1, 0.1))
        solution = @suppress plastic_shear(igg; observation, λ_relaxation, adjoint = true, cells)
        @test solution.yielded > 0
        fd = central_difference(0.25, 1.0e-5) do C_matrix
            @suppress plastic_shear(igg; observation, C_matrix, λ_relaxation, cells).J
        end
        @test solution.gradient ≈ fd rtol = 1.0e-6
    end
    finalize_global_grid(; finalize_MPI = false)
end
