push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
import LinearAlgebra: dot

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

# Internal kernels and pullbacks live in the module of the selected backend.
const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax2D
end

const TA = PTArray(backend)

# `DYREL` reduces over MPI; the grid size varies between cases, so no global grid is set up.
JustRelax.MPI.Initialized() || JustRelax.MPI.Init()

# Each test checks that a pullback is the transpose of the linearized forward map,
# ⟨J v, w⟩ = ⟨v, Jᵀ w⟩, for pseudo-random directions v and seeds w. J v is a central
# difference of the production forward kernels; Jᵀ w comes from the adjoint wrappers.

# Deterministic pseudo-random values in (-1, 1); the test environment has no `Random`.
function pseudo_random(dims, seed)
    values = [2 * mod(sin(12.9898 * k + 78.233 * seed) * 43758.5453, 1.0) - 1 for k in 1:prod(dims)]
    return reshape(values, dims)
end

# Every floating-point array of a solver state, to restore it between forward evaluations.
state_arrays!(out, x::AbstractArray{<:AbstractFloat}) = push!(out, x)
state_arrays!(out, x::Union{Tuple, NamedTuple}) = (foreach(y -> state_arrays!(out, y), x); out)
function state_arrays!(out, x)
    parentmodule(typeof(x)) === JustRelax || return out
    foreach(name -> state_arrays!(out, getfield(x, name)), fieldnames(typeof(x)))
    return out
end
state_arrays(x...) = unique(objectid, state_arrays!(Any[], x))

"""
    dot_product_test(forward!, reverse!, inputs, outputs, state; masks, h)

`inputs` and `outputs` are tuples of `primal => adjoint` array pairs; an in-place update lists
its array in both. `forward!()` evaluates the forward map from the primals in `state`;
`reverse!()` consumes the output seeds and accumulates into the input adjoints. The masks limit
the directions and seeds to the degrees of freedom of a reduced system.
"""
function dot_product_test(
        forward!, reverse!, inputs, outputs, state, adjoint_state;
        input_masks = map(_ -> nothing, inputs), output_masks = map(_ -> nothing, outputs),
        h = 1.0e-6, seed = 1,
    )
    # Linearize about a consistent state: the reverse reads the intermediates of the chain.
    forward!()
    arrays = state_arrays(state...)
    saved = map(copy, arrays)
    restore!() = foreach(copyto!, arrays, saved)
    restrict(A, ::Nothing) = A
    restrict(A, mask) = A .* mask
    v = map((p, m, k) -> restrict(pseudo_random(size(first(p)), seed + k), m), inputs, input_masks, Tuple(eachindex(inputs)))
    w = map((p, m, k) -> restrict(pseudo_random(size(first(p)), seed + 100 + k), m), outputs, output_masks, Tuple(eachindex(outputs)))

    function response(step)
        restore!()
        foreach((p, d) -> first(p) .+= step .* TA(d), inputs, v)
        forward!()
        return map(p -> Array(first(p)), outputs)
    end
    plus, minus = response(h), response(-h)
    Jv_w = sum(map((a, b, c) -> dot(a - b, c), plus, minus, w)) / 2h

    restore!()
    foreach(A -> fill!(A, 0), state_arrays(adjoint_state...))
    foreach((p, d) -> copyto!(last(p), TA(d)), outputs, w)
    reverse!()
    v_JTw = sum(map((p, d) -> dot(d, Array(last(p))), inputs, v))
    restore!()
    return Jv_w, v_JTw
end

@parallel_indices (i, j) function init_two_phases!(phases, offset)
    f = 0.5 + 0.4 * sin(1.3 * i + 0.7 * j + offset)
    @index phases[1, i, j] = f
    @index phases[2, i, j] = 1 - f
    return nothing
end

# Phase 2 is air above y = 0.7, with a mixed layer, so the rock ratio masks some rows.
@parallel_indices (i, j) function init_air!(phases, y)
    air = clamp((y[j] - 0.6) / 0.2, 0.0, 1.0)
    @index phases[1, i, j] = 1 - air
    @index phases[2, i, j] = air
    return nothing
end

function two_phase_ratios(ni; air = nothing)
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    if isnothing(air)
        @parallel (@idx ni) init_two_phases!(phase_ratios.center, 0.0)
        @parallel (@idx ni .+ 1) init_two_phases!(phase_ratios.vertex, 0.3)
    else
        @parallel (@idx ni) init_air!(phase_ratios.center, air.xci[2])
        @parallel (@idx ni .+ 1) init_air!(phase_ratios.vertex, air.xvi[2])
    end
    return phase_ratios
end

function material(phase; plasticity = (), G = 1.0, η = 1.0)
    elasticity = ConstantElasticity(; G, Kb = 5.0)
    return SetMaterialParams(;
        Phase = phase,
        Density = PT_Density(; ρ0 = 1.0 + 0.2 * phase, β = 0.05, α = 0.0, T0 = 0.0),
        Gravity = ConstantGravity(; g = 1.0),
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((LinearViscous(; η), elasticity, plasticity...)),
    )
end

const rheologies = (
    viscoelastic = (material(1), material(2; G = 0.5, η = 3.0)),
    drucker_prager = (
        material(1; plasticity = (DruckerPrager_regularised(; C = 0.05, ϕ = 30.0, Ψ = 10.0, η_vp = 0.1),)),
        material(2; G = 0.5, plasticity = (DruckerPrager_regularised(; C = 0.08, ϕ = 25.0, Ψ = 5.0, η_vp = 0.2),)),
    ),
    cap = (
        material(1; plasticity = (DruckerPragerCap(; C = 0.05, ϕ = 30.0, Ψ = 0.0, η_vp = 0.1, pT = -0.05),)),
        material(2; G = 0.5),
    ),
)

const flow_bcs_cases = (
    free_slip = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    ),
    no_slip = VelocityBoundaryConditions(;
        no_slip = (left = true, right = true, top = true, bot = true),
        free_slip = (left = false, right = false, top = false, bot = false),
    ),
    periodic_x = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
        periodic = (left = true, right = true, top = false, bot = false),
    ),
)

# A forward state with O(1) strain rates, so that the plastic cases yield.
function setup(ni, bcs, rheology; air = false)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    stokes = StokesArrays(backend, ni, bcs)
    adjoint = AdjointStokesArrays(backend, ni, bcs)
    phase_ratios = two_phase_ratios(ni; air = air ? grid : nothing)
    fields = (
        stokes.V.Vx, stokes.V.Vy, stokes.P, stokes.ΔPψ, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy,
        stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c, stokes.τ.xx_v, stokes.τ.yy_v,
        stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy,
    )
    for (k, A) in enumerate(fields)
        copyto!(A, TA(pseudo_random(size(A), k)))
    end
    stokes.P .= 0.1 .+ 0.15 .* stokes.P
    stokes.viscosity.η .= 1.0 .+ 0.5 .* TA(pseudo_random(size(stokes.viscosity.η), 50)) .^ 2
    stokes.viscosity.ηv .= 1.0 .+ 0.5 .* TA(pseudo_random(size(stokes.viscosity.ηv), 51)) .^ 2
    ρg = (TA(pseudo_random(ni, 60)), 1.0 .+ 0.1 .* TA(pseudo_random(ni, 61)))
    dt = 0.5
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt)
    ϕ = if air
        rock = RockRatio(backend, ni)
        update_rock_ratio!(rock, phase_ratios, 2)
        rock
    else
        nothing
    end
    dyrel = isnothing(ϕ) ?
        DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt) :
        DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt)
    return (; grid, stokes, adjoint, phase_ratios, ρg, dt, args, ϕ, dyrel, rheology, bcs, ni)
end

# Threads compiling the same deferred Enzyme gradient at once used to deadlock the pointwise
# reverse. The test process may run with one thread, so this starts its own with two.
@testset "Pointwise reverse with two threads" begin
    if backend === JustRelax.CPUBackend
        code = """
        using JustRelax, JustRelax.JustRelax2D, ParallelStencil
        @init_parallel_stencil(Threads, Float64, 2)
        square!(y, x, i, j) = (y[i, j] = x[i, j]^2; nothing)
        ni = (8, 6)
        x, x̄ = rand(ni...), zeros(ni...)
        JustRelax2D.enzyme_reverse_pointwise!(square!, ni, (zeros(ni...) => ones(ni...), x => x̄))
        exit(x̄ ≈ 2 .* x ? 0 : 1)
        """
        command = `$(Base.julia_cmd()) -t 2 --startup-file=no --project=$(Base.active_project()) -e $code`
        process = run(pipeline(command; stdout = devnull, stderr = devnull); wait = false)
        finished = timedwait(() -> process_exited(process), 600.0) === :ok
        finished || kill(process)
        @test finished
        @test finished && success(process)
    end
end

@testset "Adjoint pullbacks are transposes of the forward kernels" begin
    sizes = ((5, 4), (4, 5))

    @testset "momentum residual, $name, ni = $ni" for (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheologies.viscoelastic)
        (; stokes, adjoint, ρg, grid) = s
        free_surface_dt = 0.3
        forward!() = @parallel (@idx ni) JR2K.compute_PH_residual_V!(
            stokes.R.Rx, stokes.R.Ry, @velocity(stokes)..., stokes.P, stokes.ΔPψ,
            @stress(stokes)..., ρg..., grid._di.center, grid._di.vertex, free_surface_dt,
        )
        reverse!() = JR2K.enzyme_compute_PH_residual_V!(
            stokes, adjoint, ρg, grid._di, ni; free_surface_dt
        )
        inputs = (
            stokes.V.Vx => adjoint.V.Vx, stokes.V.Vy => adjoint.V.Vy, stokes.P => adjoint.P,
            stokes.ΔPψ => adjoint.θ, stokes.τ.xx => adjoint.τ.xx, stokes.τ.yy => adjoint.τ.yy,
            stokes.τ.xy => adjoint.τ.xy, ρg[1] => adjoint.dρgx, ρg[2] => adjoint.ρ,
        )
        outputs = (stokes.R.Rx => adjoint.R.Rx, stokes.R.Ry => adjoint.R.Ry)
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes, ρg), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-7
    end

    @testset "continuity and strain rate, $name, ni = $ni" for (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheologies.viscoelastic)
        (; stokes, adjoint, dyrel, rheology, phase_ratios, grid, dt, args) = s
        forward!() = JR2K.compute_∇V_strain_rate_RP!(
            stokes, dyrel, rheology, phase_ratios, grid._di, ni, dt, true; args...
        )
        reverse!() = JR2K.enzyme_compute_∇V_strain_rate_RP!(
            stokes, adjoint, dyrel, rheology, phase_ratios, grid._di, ni, dt, args
        )
        inputs = (stokes.V.Vx => adjoint.V.Vx, stokes.V.Vy => adjoint.V.Vy, stokes.P => adjoint.P)
        outputs = (
            stokes.ε.xx => adjoint.ε.xx, stokes.ε.yy => adjoint.ε.yy,
            stokes.ε.xy => adjoint.ε.xy, stokes.R.RP => adjoint.R.RP,
        )
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes,), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-7
    end

    @testset "stress, $rheology_name, $name, ni = $ni" for (rheology_name, rheology) in pairs(rheologies),
            (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheology)
        (; stokes, adjoint, phase_ratios, dt, args) = s
        forward!() = JR2K.compute_stress_DRYEL!(
            stokes, rheology, phase_ratios, 1.0, dt; Pf = JR2K.fluid_pressure(args, stokes.P)
        )
        reverse!() = JR2K.enzyme_reverse_stress_DRYEL!(
            stokes, rheology, phase_ratios, 1.0, dt, args;
            τ_bar = (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
            τv_bar = (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy),
            τII_bar = adjoint.τ.II,
            ε_bar = (adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy),
            P_bar = adjoint.P, η_bar = adjoint.viscosity.η, ηv_bar = adjoint.viscosity.ηv,
            θ_bar = adjoint.θ,
        )
        inputs = (
            stokes.ε.xx => adjoint.ε.xx, stokes.ε.yy => adjoint.ε.yy, stokes.ε.xy => adjoint.ε.xy,
            stokes.P => adjoint.P, stokes.viscosity.η => adjoint.viscosity.η,
            stokes.viscosity.ηv => adjoint.viscosity.ηv,
        )
        outputs = (
            stokes.τ.xx => adjoint.τ.xx, stokes.τ.yy => adjoint.τ.yy, stokes.τ.xy_c => adjoint.τ.xy_c,
            stokes.τ.xx_v => adjoint.τ.xx_v, stokes.τ.yy_v => adjoint.τ.yy_v, stokes.τ.xy => adjoint.τ.xy,
            stokes.τ.II => adjoint.τ.II, stokes.ΔPψ => adjoint.θ,
        )
        forward!()
        rheology_name === :viscoelastic || @test any(>(0), Array(stokes.λ))
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes,), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-6
    end

    @testset "pressure-dependent buoyancy, ni = $ni" for ni in sizes
        s = setup(ni, flow_bcs_cases.free_slip, rheologies.viscoelastic)
        (; stokes, adjoint, phase_ratios, ρg, args, rheology) = s
        forward!() = JR2K.update_ρg!(ρg, phase_ratios, rheology, args)
        reverse!() = JR2K.enzyme_compute_ρg!(ρg, adjoint, phase_ratios, rheology, args, ni)
        inputs = (stokes.P => adjoint.P,)
        outputs = (ρg[1] => adjoint.dρgx, ρg[2] => adjoint.ρ)
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes, ρg), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-7
    end

    @testset "power-law viscosity, ni = $ni" for ni in sizes
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((PowerlawViscous(; η0 = 2.0, n = 3.0, ε0 = 1.0),)),
            ),
            SetMaterialParams(;
                Phase = 2,
                CompositeRheology = CompositeRheology((PowerlawViscous(; η0 = 1.0, n = 2.0, ε0 = 1.0),)),
            ),
        )
        s = setup(ni, flow_bcs_cases.free_slip, rheologies.viscoelastic)
        (; stokes, adjoint, phase_ratios, args) = s
        cutoff = (-Inf, Inf)
        forward!() = compute_viscosity_τII!(stokes, phase_ratios, args, rheology, cutoff)
        reverse!() = JR2K.enzyme_compute_viscosity_τII!(
            (stokes.viscosity.η, stokes.viscosity.ηv),
            (adjoint.viscosity.η, adjoint.viscosity.ηv),
            ((adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c), (adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy)),
            stokes, phase_ratios, args, rheology, cutoff,
        )
        inputs = (
            stokes.τ.xx => adjoint.τ.xx, stokes.τ.yy => adjoint.τ.yy, stokes.τ.xy_c => adjoint.τ.xy_c,
            stokes.τ.xx_v => adjoint.τ.xx_v, stokes.τ.yy_v => adjoint.τ.yy_v, stokes.τ.xy => adjoint.τ.xy,
        )
        outputs = (stokes.viscosity.η => adjoint.viscosity.η, stokes.viscosity.ηv => adjoint.viscosity.ηv)
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes,), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-7
    end

    @testset "velocity boundary conditions, $name, ni = $ni" for (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheologies.viscoelastic)
        (; stokes, adjoint) = s
        forward!() = flow_bcs!(stokes, bcs)
        reverse!() = JR2K.enzyme_flow_bcs!(stokes, adjoint, bcs)
        velocities = (stokes.V.Vx => adjoint.V.Vx, stokes.V.Vy => adjoint.V.Vy)
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, velocities, velocities, (stokes,), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-7
    end

    # One sweep of `adjoint_residual!` is the transpose of the linearized forward residual
    # chain: boundary conditions, buoyancy, strain rate and continuity, stress, momentum.
    @testset "adjoint residual, $rheology_name, $name, ni = $ni" for (rheology_name, rheology) in pairs(rheologies),
            (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheology)
        (; stokes, adjoint, phase_ratios, ρg, dyrel, grid, dt, args) = s
        free_surface = true
        function forward!()
            flow_bcs!(stokes, bcs)
            JR2K.update_ρg!(ρg, phase_ratios, rheology, args)
            JR2K.compute_∇V_strain_rate_RP!(
                stokes, dyrel, rheology, phase_ratios, grid._di, ni, dt, true; args...
            )
            JR2K.compute_stress_DRYEL!(
                stokes, rheology, phase_ratios, 1.0, dt; Pf = JR2K.fluid_pressure(args, stokes.P)
            )
            @parallel (@idx ni) JR2K.compute_PH_residual_V!(
                stokes.R.Rx, stokes.R.Ry, @velocity(stokes)..., stokes.P, stokes.ΔPψ,
                @stress(stokes)..., ρg..., grid._di.center, grid._di.vertex, dt * free_surface,
            )
            return nothing
        end
        function reverse!()
            # `initialize_adjoint_iteration!` moves the λ seeds into the residual adjoints.
            Rx, Ry = size(stokes.R.Rx), size(stokes.R.Ry)
            adjoint.λV.Vx[2:(Rx[1] + 1), 2:(Rx[2] + 1)] .= adjoint.R.Rx
            adjoint.λV.Vy[2:(Ry[1] + 1), 2:(Ry[2] + 1)] .= adjoint.R.Ry
            adjoint.λP .= adjoint.R.RP
            JR2K.initialize_adjoint_iteration!(adjoint, ni)
            JR2K.adjoint_residual!(
                stokes, adjoint, ρg, dyrel, bcs, phase_ratios, rheology, args, grid._di, ni,
                dt, 1.0, free_surface, (-Inf, Inf), true, nothing,
            )
            return nothing
        end
        inputs = (stokes.V.Vx => adjoint.V.Vx, stokes.V.Vy => adjoint.V.Vy, stokes.P => adjoint.P)
        outputs = (stokes.R.Rx => adjoint.R.Rx, stokes.R.Ry => adjoint.R.Ry, stokes.R.RP => adjoint.R.RP)
        Jv_w, v_JTw = dot_product_test(forward!, reverse!, inputs, outputs, (stokes, ρg), (adjoint,))
        @test Jv_w ≈ v_JTw rtol = 1.0e-6
    end

    # Variational chain with sticky air: directions and seeds live on the reduced (rock) space,
    # the degrees of freedom of the variational adjoint.
    @testset "variational adjoint residual, $rheology_name, $name, ni = $ni" for (rheology_name, rheology) in pairs(rheologies),
            (name, bcs) in pairs(flow_bcs_cases), ni in sizes
        s = setup(ni, bcs, rheology; air = true)
        (; stokes, adjoint, phase_ratios, ρg, dyrel, grid, dt, args, ϕ) = s
        air_phase = 2
        free_surface = true
        maskV = (
            similar(ϕ.Vx, Bool, size(stokes.R.Rx)), similar(ϕ.Vy, Bool, size(stokes.R.Ry)),
        )
        maskP = similar(ϕ.center, Bool)
        @parallel (@idx ni) JR2K.update_valid_c_mask!(maskP, ϕ)
        @parallel (@idx ni) JR2K.update_valid_v_masks!(maskV..., ϕ)
        @test !all(Array(maskP)) && any(Array(maskP))
        function forward!()
            flow_bcs!(stokes, bcs)
            JR2K.update_ρg!(ρg, phase_ratios, rheology, args; air_phase)
            JR2K.compute_∇V_strain_rate_RP!(
                stokes, dyrel, rheology, phase_ratios, ϕ, grid._di, ni, dt, true; args...
            )
            JR2K.compute_stress_DRYEL!(
                stokes, rheology, phase_ratios, ϕ, 1.0, dt; Pf = JR2K.fluid_pressure(args, stokes.P)
            )
            @parallel (@idx ni) JR2K.compute_PH_residual_V!(
                stokes.R.Rx, stokes.R.Ry, @velocity(stokes)..., stokes.P, stokes.ΔPψ,
                @stress(stokes)..., ρg..., ϕ, grid._di.center, grid._di.vertex, dt * free_surface,
            )
            return nothing
        end
        function reverse!()
            Rx, Ry = size(stokes.R.Rx), size(stokes.R.Ry)
            adjoint.λV.Vx[2:(Rx[1] + 1), 2:(Rx[2] + 1)] .= adjoint.R.Rx
            adjoint.λV.Vy[2:(Ry[1] + 1), 2:(Ry[2] + 1)] .= adjoint.R.Ry
            adjoint.λP .= adjoint.R.RP
            JR2K.initialize_adjoint_iteration!(adjoint, ni)
            JR2K.variational_adjoint_residual!(
                stokes, adjoint, ρg, dyrel, bcs, phase_ratios, ϕ, rheology, args, grid._di, ni,
                dt, 1.0, free_surface, air_phase, maskP, nothing, (-Inf, Inf),
            )
            return nothing
        end
        # Velocity rows in the reduced space; the boundary and ghost entries are not unknowns.
        function velocity_mask(V, mask)
            full = zeros(Bool, size(V))
            full[2:(size(mask, 1) + 1), 2:(size(mask, 2) + 1)] .= Array(mask)
            return full
        end
        inputs = (stokes.V.Vx => adjoint.V.Vx, stokes.V.Vy => adjoint.V.Vy, stokes.P => adjoint.P)
        outputs = (stokes.R.Rx => adjoint.R.Rx, stokes.R.Ry => adjoint.R.Ry, stokes.R.RP => adjoint.R.RP)
        Jv_w, v_JTw = dot_product_test(
            forward!, reverse!, inputs, outputs, (stokes, ρg), (adjoint,);
            input_masks = (velocity_mask(stokes.V.Vx, maskV[1]), velocity_mask(stokes.V.Vy, maskV[2]), Array(maskP)),
            output_masks = (Array(maskV[1]), Array(maskV[2]), Array(maskP)),
        )
        @test Jv_w ≈ v_JTw rtol = 1.0e-6
    end
    # The per-iteration reset (`reset_adjoint_residuals!`) zeroes only P and V: one reverse sweep
    # must consume every stress, strain-rate and residual seed.
    @testset "sweep consumes its seeds, $(variational ? "variational" : "standard"), $rheology_name, $name" for variational in (false, true),
            (rheology_name, rheology) in pairs(rheologies), (name, bcs) in pairs(flow_bcs_cases)
        ni = (5, 4)
        s = setup(ni, bcs, rheology; air = variational)
        (; stokes, adjoint, phase_ratios, ρg, dyrel, grid, dt, args, ϕ) = s
        adjoint.λV.Vx .= TA(pseudo_random(size(adjoint.λV.Vx), 7))
        adjoint.λV.Vy .= TA(pseudo_random(size(adjoint.λV.Vy), 8))
        adjoint.λP .= TA(pseudo_random(ni, 9))
        JR2K.initialize_adjoint_iteration!(adjoint, ni)
        if variational
            maskP = similar(ϕ.center, Bool)
            @parallel (@idx ni) JR2K.update_valid_c_mask!(maskP, ϕ)
            JR2K.variational_adjoint_residual!(
                stokes, adjoint, ρg, dyrel, bcs, phase_ratios, ϕ, rheology, args, grid._di, ni,
                dt, 1.0, true, 2, maskP, nothing, (-Inf, Inf),
            )
        else
            JR2K.adjoint_residual!(
                stokes, adjoint, ρg, dyrel, bcs, phase_ratios, rheology, args, grid._di, ni,
                dt, 1.0, true, (-Inf, Inf), true, nothing,
            )
        end
        seeds = (
            adjoint.ε.xx, adjoint.ε.yy, adjoint.ε.xy, adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c,
            adjoint.τ.II, adjoint.τ.xx_v, adjoint.τ.yy_v, adjoint.τ.xy, adjoint.θ,
            adjoint.R.Rx, adjoint.R.Ry, adjoint.R.RP,
        )
        @test all(A -> all(iszero, Array(A)), seeds)
    end
end
