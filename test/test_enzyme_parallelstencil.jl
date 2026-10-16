get!(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax3D as JR3
using JustPIC
using ParallelStencil

@init_parallel_stencil(Threads, Float64, 2)

const Enzyme = JustRelax2D.Enzyme

@parallel_indices (i, j) function _init_enzyme_phase!(phase)
    @index phase[1, i, j] = 1.0
    return nothing
end

@testset "Variational DYREL stress-viscosity pullback" begin
    ni = (3, 2)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    phases = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phases.vertex)
    ϕ = RockRatio(CPUBackend, ni)
    update_rock_ratio!(ϕ, phases, 0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((PowerlawViscous(; η0 = 2.0, n = 3.0, ε0 = 1.0),)),
        ),
    )
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.τ.xx .= 0.3
    stokes.τ.yy .= -0.1
    stokes.τ.xy_c .= 0.2
    stokes.τ.xy_c[2, 2] = 0.27
    weights = reshape(collect(1.0:6.0), ni...)

    function objective()
        JustRelax2D.update_viscosity_τII!(
            stokes, phases, ϕ, args, rheology, (-Inf, Inf); relaxation = 1.0,
        )
        return sum(weights .* stokes.viscosity.η)
    end
    objective()
    adjoint.viscosity.η .= weights
    JustRelax2D.enzyme_compute_viscosity_point!(
        copy(stokes.viscosity.η), adjoint.viscosity.η, phases.center,
        (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c),
        (adjoint.τ.xx, adjoint.τ.yy, adjoint.τ.xy_c),
        args, rheology, (-Inf, Inf), JustRelax2D.compute_viscosity_τII,
        JustRelax2D.local_viscosity_args,
    )

    function finite_difference!(field, index)
        h = 1.0e-6
        field[index...] += h
        plus = objective()
        field[index...] -= 2h
        minus = objective()
        field[index...] += h
        return (plus - minus) / (2h)
    end
    @test adjoint.τ.xx[2, 1] ≈ finite_difference!(stokes.τ.xx, (2, 1)) rtol = 1.0e-5
    @test adjoint.τ.xy_c[2, 2] ≈ finite_difference!(stokes.τ.xy_c, (2, 2)) rtol = 1.0e-5
end

@testset "Nested composite viscosity parameter pullback" begin
    ni = (2, 2)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    phases = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phases.vertex)
    make_rheology(η0) = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology(
                (
                    Parallel(
                        PowerlawViscous(; η0, n = 3.0, ε0 = 1.0),
                        LinearViscous(; η = 4.0),
                    ),
                )
            ),
        ),
    )
    rheology = make_rheology(2.0)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.τ.xx .= 0.3
    stokes.τ.yy .= -0.1
    stokes.τ.xy_c .= 0.2
    stokes.τ.xx_v .= 0.25
    stokes.τ.yy_v .= -0.15
    stokes.τ.xy .= 0.1

    function objective(test_rheology)
        JustRelax2D.compute_viscosity_τII!(
            stokes, phases, args, test_rheology, (-Inf, Inf); relaxation = 1.0,
        )
        return sum(stokes.viscosity.η) + sum(stokes.viscosity.ηv)
    end

    objective(rheology)
    adjoint.viscosity.η .= 1.0
    adjoint.viscosity.ηv .= 1.0
    gradients = material_controls(CPUBackend, ni, (:η0,); nphases = 1)
    JustRelax2D.compute_viscosity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, (-Inf, Inf), gradients,
    )

    h = 1.0e-6
    finite_difference = (objective(make_rheology(2.0 + h)) - objective(make_rheology(2.0 - h))) / (2h)
    predicted = sum(gradients.η0.center) + sum(gradients.η0.vertex)
    @test predicted ≈ finite_difference rtol = 1.0e-5

    center_objective(test_rheology) = begin
        JustRelax2D.compute_viscosity_τII!(
            stokes, phases, args, test_rheology, (-Inf, Inf); relaxation = 1.0,
        )
        return sum(stokes.viscosity.η)
    end
    gradients = material_controls(CPUBackend, ni, (:η0,); nphases = 1)
    JustRelax2D.compute_viscosity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, (-Inf, Inf), gradients;
        include_vertex = false,
    )
    center_fd = (
        center_objective(make_rheology(2.0 + h)) - center_objective(make_rheology(2.0 - h))
    ) / (2h)
    @test sum(gradients.η0.center) ≈ center_fd rtol = 1.0e-5
    @test all(iszero, gradients.η0.vertex)
end

@parallel_indices (i, j) function _init_G_gradient_phases!(phase)
    @index phase[1, i, j] = 0.2
    @index phase[2, i, j] = 0.5
    @index phase[3, i, j] = 0.3
    return nothing
end

@parallel_indices (i, j) function _init_α_gradient_phases!(phase)
    @index phase[1, i, j] = i == 1 ? 0.6 : 0.2
    @index phase[2, i, j] = i == 1 ? 0.1 : 0.5
    @index phase[3, i, j] = i == 1 ? 0.3 : 0.3
    return nothing
end

@parallel_indices (i, j) function _init_η_gradient_phases!(phase)
    @index phase[1, i, j] = i == 1 ? 1.0 : 0.35
    @index phase[2, i, j] = i == 1 ? 0.0 : 0.65
    return nothing
end

function _elastic_residual_objective!(stokes, adjoint, rheology, phases, ρg, grid, dt)
    JustRelax2D.compute_stress_DRYEL!(stokes, rheology, phases, 1.0, dt)
    @parallel (@idx size(stokes.P)) JustRelax2D.compute_PH_residual_V!(
        stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
        stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
        grid._di.center, grid._di.vertex,
    )
    return -sum(stokes.R.Rx .* @view(adjoint.λV.Vx[2:(end - 1), 2:(end - 1)])) -
        sum(stokes.R.Ry .* @view(adjoint.λV.Vy[2:(end - 1), 2:(end - 1)]))
end

function _elastic_rheology(moduli, bulk_moduli)
    return ntuple(length(moduli)) do p
        elasticity = ConstantElasticity(; G = moduli[p], Kb = bulk_moduli[p])
        plasticity = DruckerPrager_regularised(;
            ϕ = 20.0, Ψ = p == 3 ? 0.0 : 10.0, C = 0.01, η_vp = 0.1
        )
        SetMaterialParams(;
            Phase = p,
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology(
                (
                    LinearViscous(; η = 3.0), elasticity, plasticity,
                )
            ),
            Elasticity = elasticity,
            Plasticity = plasticity,
        )
    end
end

function _linear_viscosity_rheology(values)
    return ntuple(length(values)) do p
        SetMaterialParams(;
            Phase = p,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = values[p]),)),
        )
    end
end

function _plastic_rheology(parameters)
    return ntuple(length(parameters)) do p
        elasticity = ConstantElasticity(; G = 1.5, Kb = 4.0)
        plasticity = DruckerPrager_regularised(; parameters[p]...)
        SetMaterialParams(;
            Phase = p,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology(
                (
                    LinearViscous(; η = 2.0), elasticity, plasticity,
                )
            ),
            Elasticity = elasticity,
            Plasticity = plasticity,
        )
    end
end

function _linear_viscosity_residual_objective!(
        stokes, adjoint, rheology, phases, args, ρg, grid, dt
    )
    compute_viscosity!(stokes, phases, args, rheology, (-Inf, Inf))
    JustRelax2D.compute_stress_DRYEL!(stokes, rheology, phases, 1.0, dt)
    @parallel (@idx size(stokes.P)) JustRelax2D.compute_PH_residual_V!(
        stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
        stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
        grid._di.center, grid._di.vertex,
    )
    return -sum(stokes.R.Rx .* @view(adjoint.λV.Vx[2:(end - 1), 2:(end - 1)])) -
        sum(stokes.R.Ry .* @view(adjoint.λV.Vy[2:(end - 1), 2:(end - 1)]))
end

@testset "Normal DYREL nonlinear constitutive pullback" begin
    ni = (3, 3)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    phases = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phases.vertex)
    powerlaw_parameters = (; η0 = 2.0, n = 3.0, ε0 = 1.0)
    make_rheology(parameters = powerlaw_parameters) = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((PowerlawViscous(; parameters...),)),
        ),
    )
    rheology = make_rheology()
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.ε.xx .= 0.3
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.2
    center_weights = reshape(collect(1.0:9.0), ni...)
    vertex_weights = reshape(collect(1.0:16.0), (ni .+ 1)...)

    function converged_objective(test_rheology = rheology)
        stokes.viscosity.η .= 2.0
        stokes.viscosity.ηv .= 2.0
        for _ in 1:200
            JustRelax2D.compute_stress_DRYEL!(stokes, test_rheology, phases, 1.0, 1.0)
            JustRelax2D.compute_viscosity_τII!(
                stokes, phases, args, test_rheology, (-Inf, Inf); relaxation = 0.5,
            )
        end
        JustRelax2D.compute_stress_DRYEL!(stokes, test_rheology, phases, 1.0, 1.0)
        return sum(center_weights .* stokes.τ.xx) + sum(vertex_weights .* stokes.τ.xy)
    end

    converged_objective()
    adjoint.τ.xx .= center_weights
    adjoint.τ.xy .= vertex_weights
    scratch = (;
        η_target = copy(stokes.viscosity.η),
        ηv_target = copy(stokes.viscosity.ηv),
        η = similar(stokes.viscosity.η),
        ηv = similar(stokes.viscosity.ηv),
        feedback = similar(stokes.viscosity.η),
        feedback_v = similar(stokes.viscosity.ηv),
        feedback_λ = Ref{Any}(nothing),
        τ = map(similar, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)),
        τ_v = map(similar, (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)),
        τII = similar(stokes.τ.II),
        θ = similar(stokes.ΔPψ),
    )
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phases, 1.0, 1.0, args, (-Inf, Inf), false, scratch,
    )
    dεxx = adjoint.ε.xx[2, 2]
    dεxy = adjoint.ε.xy[2, 2]

    function finite_difference!(field, index)
        h = 1.0e-6
        field[index...] += h
        plus = converged_objective()
        field[index...] -= 2h
        minus = converged_objective()
        field[index...] += h
        return (plus - minus) / (2h)
    end
    @test dεxx ≈ finite_difference!(stokes.ε.xx, (2, 2)) rtol = 1.0e-5
    @test dεxy ≈ finite_difference!(stokes.ε.xy, (2, 2)) rtol = 1.0e-5

    # Material gradients use the same implicit viscosity seed as the strain-rate pullback.
    converged_objective()
    adjoint.τ.xx .= center_weights
    adjoint.τ.yy .= 0
    adjoint.τ.xy_c .= 0
    adjoint.τ.xx_v .= 0
    adjoint.τ.yy_v .= 0
    adjoint.τ.xy .= vertex_weights
    adjoint.τ.II .= 0
    adjoint.θ .= 0
    adjoint.viscosity.η .= 0
    adjoint.viscosity.ηv .= 0
    JustRelax2D.enzyme_compute_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phases, 1.0, 1.0, args, (-Inf, Inf),
    )
    gradients = material_controls(
        CPUBackend, ni, keys(powerlaw_parameters); nphases = 1
    )
    JustRelax2D.compute_viscosity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, (-Inf, Inf), gradients;
        viscosity_seed = (scratch.η, scratch.ηv),
    )
    for name in keys(powerlaw_parameters)
        h = 1.0e-6
        value = powerlaw_parameters[name]
        plus = merge(powerlaw_parameters, NamedTuple{(name,)}((value + h,)))
        minus = merge(powerlaw_parameters, NamedTuple{(name,)}((value - h,)))
        finite_difference = (
            converged_objective(make_rheology(plus)) -
                converged_objective(make_rheology(minus))
        ) / (2h)
        predicted = sum(gradients[name].center) + sum(gradients[name].vertex)
        @test predicted ≈ finite_difference rtol = 1.0e-5
    end
end

@testset "Variational DYREL nonlinear constitutive pullback" begin
    ni = (3, 3)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    phases = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phases.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phases.vertex)
    ϕ = RockRatio(CPUBackend, ni)
    update_rock_ratio!(ϕ, phases, 0)
    powerlaw_parameters = (; η0 = 2.0, n = 3.0, ε0 = 1.0)
    make_rheology(parameters = powerlaw_parameters) = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((PowerlawViscous(; parameters...),)),
        ),
    )
    rheology = make_rheology()
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.ε.xx .= 0.3
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.2
    center_weights = reshape(collect(1.0:9.0), ni...)
    vertex_weights = reshape(collect(1.0:16.0), (ni .+ 1)...)

    function converged_objective(test_rheology = rheology)
        stokes.viscosity.η .= 2.0
        for _ in 1:200
            JustRelax2D.compute_stress_DRYEL!(stokes, test_rheology, phases, ϕ, 1.0, 1.0)
            JustRelax2D.update_viscosity_τII!(
                stokes, phases, ϕ, args, test_rheology, (-Inf, Inf); relaxation = 0.5,
            )
        end
        JustRelax2D.compute_stress_DRYEL!(stokes, test_rheology, phases, ϕ, 1.0, 1.0)
        return sum(center_weights .* stokes.τ.xx) + sum(vertex_weights .* stokes.τ.xy)
    end

    converged_objective()
    adjoint.τ.xx .= center_weights
    adjoint.τ.xy .= vertex_weights
    scratch = (;
        η_target = copy(stokes.viscosity.η),
        η = similar(stokes.viscosity.η),
        feedback = similar(stokes.viscosity.η),
        feedback_λ = Ref{Any}(nothing),
        τ = map(similar, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)),
        τ_v = map(similar, (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)),
        τII = similar(stokes.τ.II),
        θ = similar(stokes.ΔPψ),
    )
    JustRelax2D.enzyme_compute_variational_stress_DRYEL!(
        scratch, stokes, adjoint, rheology, phases, ϕ, 1.0, 1.0, args, (-Inf, Inf),
    )
    dεxx = adjoint.ε.xx[2, 2]
    dεxy = adjoint.ε.xy[2, 2]

    function finite_difference!(field, index)
        h = 1.0e-6
        field[index...] += h
        plus = converged_objective()
        field[index...] -= 2h
        minus = converged_objective()
        field[index...] += h
        return (plus - minus) / (2h)
    end
    @test dεxx ≈ finite_difference!(stokes.ε.xx, (2, 2)) rtol = 1.0e-5
    @test dεxy ≈ finite_difference!(stokes.ε.xy, (2, 2)) rtol = 1.0e-5

    converged_objective()
    adjoint.τ.xx .= center_weights
    adjoint.τ.yy .= 0
    adjoint.τ.xy_c .= 0
    adjoint.τ.xx_v .= 0
    adjoint.τ.yy_v .= 0
    adjoint.τ.xy .= vertex_weights
    adjoint.τ.II .= 0
    adjoint.θ .= 0
    adjoint.viscosity.η .= 0
    adjoint.viscosity.ηv .= 0
    JustRelax2D.enzyme_prepare_variational_implicit_viscosity_τII!(
        scratch, stokes, adjoint, rheology, phases, ϕ, 1.0, 1.0, args, (-Inf, Inf),
    )
    gradients = material_controls(
        CPUBackend, ni, keys(powerlaw_parameters); nphases = 1
    )
    JustRelax2D.compute_viscosity_parameter_gradients!(
        stokes, adjoint, phases, rheology, args, (-Inf, Inf), gradients;
        viscosity_seed = (scratch.η, adjoint.viscosity.ηv),
    )
    for name in keys(powerlaw_parameters)
        h = 1.0e-6
        value = powerlaw_parameters[name]
        plus = merge(powerlaw_parameters, NamedTuple{(name,)}((value + h,)))
        minus = merge(powerlaw_parameters, NamedTuple{(name,)}((value - h,)))
        finite_difference = (
            converged_objective(make_rheology(plus)) -
                converged_objective(make_rheology(minus))
        ) / (2h)
        predicted = sum(gradients[name].center) + sum(gradients[name].vertex)
        @test predicted ≈ finite_difference rtol = 1.0e-5
    end
end

@testset "Spatial elastic-modulus sensitivities" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    dt = 0.7
    moduli = (2.0, 4.0, Inf)
    bulk_moduli = (7.0, 11.0, Inf)
    rheology = _elastic_rheology(moduli, bulk_moduli)
    phases = PhaseRatios(JustPIC.CPU, 3, ni)
    @parallel (@idx ni) _init_G_gradient_phases!(phases.center)
    @parallel (@idx ni .+ 1) _init_G_gradient_phases!(phases.vertex)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    gradients = material_controls(
        CPUBackend,
        ni,
        (:G, :Kb);
        nphases = length(rheology),
    )
    stokes.viscosity.η .= 3.0
    stokes.viscosity.ηv .= 3.0
    stokes.ε.xx .= 0.2
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.3
    # Nonzero, fixed stress history must contribute to dτ/dG as well.
    stokes.τ_o.xx .= 0.4
    stokes.τ_o.yy .= -0.2
    stokes.τ_o.xy_c .= 0.1
    stokes.τ_o.xx_v .= 0.4
    stokes.τ_o.yy_v .= -0.2
    stokes.τ_o.xy .= 0.1
    adjoint.λV.Vx .= reshape(sin.(1:length(adjoint.λV.Vx)), size(adjoint.λV.Vx))
    adjoint.λV.Vy .= reshape(cos.(1:length(adjoint.λV.Vy)), size(adjoint.λV.Vy))

    # Check each phase-wise material derivative against a perturbation of the actual
    # GeoParams modulus. Repeat after poisoning the buffers to ensure each call replaces it.
    for _ in 1:2
        for gradient in values(gradients)
            gradient.center .= 123.0
            gradient.vertex .= 123.0
        end
        baseline = _elastic_residual_objective!(
            stokes, adjoint, rheology, phases, ρg, grid, dt
        )
        @test isfinite(baseline)
        JustRelax2D.compute_sensitivities!(
            stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
            (; me = 0), gradients,
        )
        for (name, modulus_values) in ((:G, moduli), (:Kb, bulk_moduli))
            gradient = gradients[name]
            @test size(gradient.center) == (length(rheology), ni...)
            @test size(gradient.vertex) == (length(rheology), (ni .+ 1)...)
            @test all(isfinite, gradient.center)
            @test all(isfinite, gradient.vertex)
            @test any(!iszero, gradient.center)

            for p in eachindex(modulus_values)
                isfinite(modulus_values[p]) || continue
                h = 1.0e-5
                plus_values = Base.setindex(modulus_values, modulus_values[p] + h, p)
                minus_values = Base.setindex(modulus_values, modulus_values[p] - h, p)
                plus_rheology = name === :G ?
                    _elastic_rheology(plus_values, bulk_moduli) :
                    _elastic_rheology(moduli, plus_values)
                minus_rheology = name === :G ?
                    _elastic_rheology(minus_values, bulk_moduli) :
                    _elastic_rheology(moduli, minus_values)
                plus = _elastic_residual_objective!(
                    stokes, adjoint, plus_rheology, phases, ρg, grid, dt
                )
                minus = _elastic_residual_objective!(
                    stokes, adjoint, minus_rheology, phases, ρg, grid, dt
                )
                predicted = sum(@view gradient.center[p, :, :])
                @test predicted ≈ (plus - minus) / (2h) rtol = 1.0e-5 atol = 1.0e-8
            end
        end
    end
end

@testset "Spatial linear-viscosity sensitivities" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    dt = 0.7
    values = (2.0, 5.0)
    rheology = _linear_viscosity_rheology(values)
    phases = PhaseRatios(JustPIC.CPU, 2, ni)
    @parallel (@idx ni) _init_η_gradient_phases!(phases.center)
    @parallel (@idx ni .+ 1) _init_η_gradient_phases!(phases.vertex)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    args = (; T = @zeros(ni .+ 2...), P = stokes.P)
    gradients = material_controls(
        CPUBackend, ni, (:η,); nphases = length(rheology)
    )
    stokes.ε.xx .= 0.2
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.3
    adjoint.λV.Vx .= reshape(sin.(1:length(adjoint.λV.Vx)), size(adjoint.λV.Vx))
    adjoint.λV.Vy .= reshape(cos.(1:length(adjoint.λV.Vy)), size(adjoint.λV.Vy))

    _linear_viscosity_residual_objective!(
        stokes, adjoint, rheology, phases, args, ρg, grid, dt
    )
    JustRelax2D.compute_sensitivities!(
        stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
        (; me = 0), gradients, args,
    )
    @test all(isfinite, gradients.η.center)
    @test all(isfinite, gradients.η.vertex)
    @test any(!iszero, gradients.η.center)

    h = 1.0e-6
    for p in eachindex(values)
        plus_values = Base.setindex(values, values[p] + h, p)
        minus_values = Base.setindex(values, values[p] - h, p)
        plus = _linear_viscosity_residual_objective!(
            stokes, adjoint, _linear_viscosity_rheology(plus_values), phases,
            args, ρg, grid, dt,
        )
        minus = _linear_viscosity_residual_objective!(
            stokes, adjoint, _linear_viscosity_rheology(minus_values), phases,
            args, ρg, grid, dt,
        )
        @test sum(gradients.η.center[p, :, :]) ≈ (plus - minus) / (2h) rtol = 1.0e-6 atol = 1.0e-8
    end
end

@testset "Spatial plastic-parameter sensitivities" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    dt = 0.5
    parameters = (
        (; C = 0.15, ϕ = 25.0, Ψ = 8.0, η_vp = 0.07),
        (; C = 0.1, ϕ = 20.0, Ψ = 5.0, η_vp = 0.05),
    )
    rheology = _plastic_rheology(parameters)
    phases = PhaseRatios(JustPIC.CPU, 2, ni)
    @parallel (@idx ni) _init_η_gradient_phases!(phases.center)
    @parallel (@idx ni .+ 1) _init_η_gradient_phases!(phases.vertex)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    requested = keys(parameters[1])
    gradients = material_controls(
        CPUBackend, ni, requested; nphases = length(rheology)
    )
    stokes.P .= 0.2
    stokes.viscosity.η .= 2.0
    stokes.viscosity.ηv .= 2.0
    stokes.ε.xx .= 0.8
    stokes.ε.yy .= -0.4
    stokes.ε.xy .= 0.5
    stokes.τ_o.xx .= 0.1
    stokes.τ_o.yy .= -0.05
    stokes.τ_o.xy_c .= 0.03
    stokes.τ_o.xx_v .= 0.1
    stokes.τ_o.yy_v .= -0.05
    stokes.τ_o.xy .= 0.03
    adjoint.λV.Vx .= reshape(sin.(1:length(adjoint.λV.Vx)), size(adjoint.λV.Vx))
    adjoint.λV.Vy .= reshape(cos.(1:length(adjoint.λV.Vy)), size(adjoint.λV.Vy))

    baseline = _elastic_residual_objective!(
        stokes, adjoint, rheology, phases, ρg, grid, dt
    )
    @test isfinite(baseline)
    @test any(>(0), stokes.λ)
    JustRelax2D.compute_sensitivities!(
        stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
        (; me = 0), gradients,
    )
    for gradient in values(gradients)
        @test all(isfinite, gradient.center)
        @test all(isfinite, gradient.vertex)
        @test any(!iszero, gradient.center)
    end

    for p in eachindex(parameters), name in keys(parameters[p])
        h = name in (:ϕ, :Ψ) ? 1.0e-5 : 1.0e-6
        plus_phase = merge(
            parameters[p], NamedTuple{(name,)}((parameters[p][name] + h,))
        )
        minus_phase = merge(
            parameters[p], NamedTuple{(name,)}((parameters[p][name] - h,))
        )
        plus = _elastic_residual_objective!(
            stokes, adjoint,
            _plastic_rheology(Base.setindex(parameters, plus_phase, p)),
            phases, ρg, grid, dt,
        )
        minus = _elastic_residual_objective!(
            stokes, adjoint,
            _plastic_rheology(Base.setindex(parameters, minus_phase, p)),
            phases, ρg, grid, dt,
        )
        @test sum(gradients[name].center[p, :, :]) ≈
            (plus - minus) / (2h) rtol = 1.0e-5 atol = 1.0e-8
    end

    # Elastic and plastic parameters share one material reverse pass.
    C_gradient = copy(gradients.C.center)
    mixed_gradients = material_controls(
        CPUBackend, ni, (:G, :C); nphases = length(rheology)
    )
    _elastic_residual_objective!(
        stokes, adjoint, rheology, phases, ρg, grid, dt
    )
    JustRelax2D.compute_sensitivities!(
        stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
        (; me = 0), mixed_gradients,
    )
    @test mixed_gradients.C.center ≈ C_gradient
    @test any(!iszero, mixed_gradients.G.center)
end

@testset "Spatial GeoParams density and gravity gradients" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    dt = 2.0
    parameters = (
        (; ρ0 = 2.0, α = 0.1, β = 0.2, T0 = 1.0, P0 = 0.3),
        (; ρ0 = 3.0, α = 0.0, T0 = 0.5),
        (; ρ = 4.0),
    )
    model_types = (PT_Density, T_Density, ConstantDensity)
    models = ntuple(p -> model_types[p](; parameters[p]...), 3)
    make_rheology(gravity = 1.0) = ntuple(3) do p
        SetMaterialParams(;
            Phase = p,
            Density = models[p],
            Gravity = ConstantGravity(; g = p == 1 ? gravity : 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        )
    end
    rheology = make_rheology()
    phases = PhaseRatios(JustPIC.CPU, 3, ni)
    @parallel (@idx ni) _init_α_gradient_phases!(phases.center)
    @parallel (@idx ni .+ 1) _init_α_gradient_phases!(phases.vertex)
    names = (:ρ0, :α, :β, :T0, :P0, :ρ, :g)
    requested = names
    gradients = material_controls(CPUBackend, ni, requested; nphases = 3)
    @test keys(gradients) == names
    @test all(
        g -> keys(g) == (:center, :vertex) &&
            size(g.center) == (3, ni...) && size(g.vertex) == (3, (ni .+ 1)...),
        values(gradients),
    )
    @test_throws ArgumentError material_controls(CPUBackend, ni, (:α,))
    @test_throws ArgumentError material_controls(
        CPUBackend, ni, (:α,); nphases = 0
    )

    ρg = (@zeros(ni...), @zeros(ni...))
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    dyrel = DYREL(CPUBackend, ni)
    dyrel.ηb .= 1.0
    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    stokes.P .= 0.7
    adjoint.λV.Vx .= reshape(sin.(1:length(adjoint.λV.Vx)), size(adjoint.λV.Vx))
    adjoint.λV.Vy .= reshape(cos.(1:length(adjoint.λV.Vy)), size(adjoint.λV.Vy))
    adjoint.λP .= reshape(sin.(1:length(adjoint.λP)), size(adjoint.λP))
    temperature = reshape(collect(1.0:20.0), ni .+ 2) ./ 10
    for thermal in (false, true)
        args = (; T = temperature, P = stokes.P)
        thermal && (args = merge(args, (; ΔT = fill(0.8, ni .+ 2))))
        # A local perturbation replaces just one phase's model at one cell.
        # The finite difference uses the original GeoParams calls, not the AD wrapper.
        objective = function (
                p = 1, I = CartesianIndex(1, 1), perturbed = models[p];
                test_rheology = rheology,
            )
            compute_ρg!(ρg, phases, test_rheology, args)
            JustRelax2D.compute_∇V_strain_rate_RP!(
                stokes, dyrel, test_rheology, phases, grid._di, ni, dt, false; args...
            )
            local_args = JustRelax2D.getindex_NamedTuple(args, Tuple(I)...)
            ratio = phases.center[I][p]
            ρg[2][I] += ratio * (
                compute_density(perturbed, local_args) - compute_density(models[p], local_args)
            )
            if thermal
                stokes.R.RP[I] += ratio * (
                    JustRelax2D.get_thermal_expansion(perturbed) -
                        JustRelax2D.get_thermal_expansion(models[p])
                ) * args.ΔT[Tuple(I + CartesianIndex(1, 1))...] / dt
            end
            @parallel (@idx ni) JustRelax2D.compute_PH_residual_V!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
                stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
                grid._di.center, grid._di.vertex,
            )
            return -sum(stokes.R.Rx .* @view(adjoint.λV.Vx[2:(end - 1), 2:(end - 1)])) -
                sum(stokes.R.Ry .* @view(adjoint.λV.Vy[2:(end - 1), 2:(end - 1)])) -
                sum(stokes.R.RP .* adjoint.λP)
        end
        objective()
        JustRelax2D.compute_sensitivities!(
            stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
            (; me = 0), gradients, args; dyrel,
        )
        density_gradient = copy(adjoint.ρ)
        for p in eachindex(models), name in Base.front(names), I in CartesianIndices(stokes.P)
            if haskey(parameters[p], name)
                h = 1.0e-6
                plus = merge(parameters[p], NamedTuple{(name,)}((parameters[p][name] + h,)))
                minus = merge(parameters[p], NamedTuple{(name,)}((parameters[p][name] - h,)))
                make_model = model_types[p]
                fd = (objective(p, I, make_model(; plus...)) - objective(p, I, make_model(; minus...))) / (2h)
                @test gradients[name].center[p, Tuple(I)...] ≈ fd rtol = 1.0e-6 atol = 1.0e-8
            else
                @test iszero(gradients[name].center[p, Tuple(I)...])
            end
        end
        h = 1.0e-6
        gravity_fd = (
            objective(; test_rheology = make_rheology(1.0 + h)) -
                objective(; test_rheology = make_rheology(1.0 - h))
        ) / (2h)
        @test sum(gradients.g.center[1, :, :]) ≈ gravity_fd rtol = 1.0e-6 atol = 1.0e-8
        @test all(iszero, gradients.g.center[2:3, :, :])
        # The density VJP must remain available; material differentiation must not consume it.
        @test adjoint.ρ == density_gradient
        saved = deepcopy(gradients)
        for g in values(gradients)
            g.center .= 123.0
        end
        objective()
        JustRelax2D.compute_sensitivities!(
            stokes, adjoint, ρg, phases, rheology, grid._di, ni, 1.0, dt,
            (; me = 0), gradients, args; dyrel,
        )
        @test all(name -> gradients[name].center ≈ saved[name].center, names)
    end

    thermal_args = (; T = temperature, P = stokes.P, ΔT = fill(0.8, ni .+ 2))
    normal_thermal = material_controls(CPUBackend, ni, (:α,); nphases = 3)
    variational_thermal = material_controls(CPUBackend, ni, (:α,); nphases = 3)
    JustRelax2D.compute_thermal_expansion_parameter_gradients!(
        stokes, adjoint, dyrel, phases, rheology, thermal_args, dt, normal_thermal,
        grid._di,
    )
    ϕ = RockRatio(CPUBackend, ni)
    update_rock_ratio!(ϕ, phases, 0)
    ϕ.center .= reshape([0.0, 0.25, 0.5, 0.75, 1.0, 0.4], ni)
    JustRelax2D.compute_thermal_expansion_parameter_gradients!(
        stokes, adjoint, dyrel, phases, rheology, thermal_args, dt, variational_thermal,
        grid._di, ϕ,
    )
    @test variational_thermal.α.center ≈
        normal_thermal.α.center .* reshape(ϕ.center, 1, ni...)
end

# Minimal working example for reverse-mode differentiation of
# `compute_∇V_strain_rate_RP!` with Enzyme and ParallelStencil.
#
# Run from the repository root with:
#
#     JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no \
#         test/test_enzyme_parallelstencil.jl
@testset "Enzyme + ParallelStencil strain-rate/RP MWE" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    (; xvi) = grid
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    dyrel = DYREL(CPUBackend, ni)

    a, b = 1.3, -0.4
    stokes.V.Vx .= [a * x for x in xvi[1], _ in 1:(ni[2] + 2)]
    stokes.V.Vy .= [b * y for _ in 1:(ni[1] + 2), y in xvi[2]]
    dyrel.ηb .= 1.0

    phase_ratios = (; center = nothing)
    # Seed L = sum(εxx) + sum(εyy) + sum(εxy) + sum(RP).
    adjoint.ε.xx .= 1.0
    adjoint.ε.yy .= 1.0
    adjoint.ε.xy .= 1.0
    adjoint.R.RP .= 1.0

    JustRelax2D.enzyme_compute_∇V_strain_rate_RP!(
        stokes,
        adjoint,
        dyrel,
        nothing,
        phase_ratios,
        grid._di,
        ni,
        1.0,
        (;),
    )

    Vx̄ = zeros(size(stokes.V.Vx))
    Vȳ = zeros(size(stokes.V.Vy))
    third = 1.0 / 3.0
    for i in 1:(ni[1] + 1), j in 1:(ni[2] + 1)
        dy_vx = JustRelax.get_dy(grid._di.velocity[1], j)
        dx_vy = JustRelax.get_dx(grid._di.velocity[2], i)
        Vx̄[i, j] -= 0.5 * dy_vx
        Vx̄[i, j + 1] += 0.5 * dy_vx
        Vȳ[i, j] -= 0.5 * dx_vy
        Vȳ[i + 1, j] += 0.5 * dx_vy

        if i ≤ ni[1] && j ≤ ni[2]
            dx, dy = JustRelax.get_dxi(grid._di.vertex, i, j)
            dVx_dx_bar = 2third - third - 1.0
            dVy_dy_bar = 2third - third - 1.0
            Vx̄[i, j + 1] -= dx * dVx_dx_bar
            Vx̄[i + 1, j + 1] += dx * dVx_dx_bar
            Vȳ[i + 1, j] -= dy * dVy_dy_bar
            Vȳ[i + 1, j + 1] += dy * dVy_dy_bar
        end
    end

    @test adjoint.V.Vx ≈ Vx̄
    @test adjoint.V.Vy ≈ Vȳ
end

@testset "Enzyme DYREL kernel wrappers" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))

    adjoint.R.Rx .= 1.0
    adjoint.R.Ry .= 1.0
    JustRelax2D.enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, grid._di, ni)
    @test any(!iszero, adjoint.P)
    @test any(!iszero, adjoint.τ.xy)
    @test any(!iszero, adjoint.dρgx)
    @test any(!iszero, adjoint.ρ)

    elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
            Elasticity = elasticity,
        ),
    )
    phase_ratios = PhaseRatios(JustPIC.CPU, 1, ni)
    @parallel (@idx ni) _init_enzyme_phase!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _init_enzyme_phase!(phase_ratios.vertex)

    stokes.viscosity.η .= 1.0
    stokes.viscosity.ηv .= 1.0
    stokes.ε.xx .= 0.2
    stokes.ε.yy .= -0.1
    stokes.ε.xy .= 0.3
    JustRelax2D.compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1.0, 1.0)
    τxx_default = copy(stokes.τ.xx)
    τxy_default = copy(stokes.τ.xy)

    θc = @zeros(ni...)
    γ_eff = @ones(ni...)
    JustRelax2D.compute_stress_viscosity_DRYEL!(
        stokes,
        θc,
        γ_eff,
        rheology,
        phase_ratios,
        1.0,
        1.0,
        1.0,
        (;),
        (-Inf, Inf),
        true,
    )
    @test stokes.τ.xx ≈ τxx_default
    @test stokes.τ.xy ≈ τxy_default

    adjoint.τ.xx .= 1.0
    adjoint.τ.yy .= 1.0
    adjoint.τ.xy .= 1.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, 1.0, 1.0,
        (;), (-Inf, Inf), true, nothing,
    )
    @test any(!iszero, adjoint.ε.xx)
    @test any(!iszero, adjoint.ε.xy)

    ϕ = RockRatio(CPUBackend, ni)
    update_rock_ratio!(ϕ, phase_ratios, 0)
    JustRelax2D.compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, 1.0, 1.0)
    adjoint.τ.xx .= 1.0
    adjoint.τ.xy .= 1.0
    adjoint.ε.xx .= 0.0
    adjoint.ε.xy .= 0.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, rheology, phase_ratios, ϕ, 1.0, 1.0, (;)
    )
    @test any(!iszero, adjoint.ε.xx)
    @test any(!iszero, adjoint.ε.xy)

    nonlinear_rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 1.0),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology(
                (
                    PowerlawViscous(; η0 = 2.0, n = 3.0, ε0 = 1.0),
                )
            ),
        ),
    )
    nonlinear_args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    stokes.viscosity.η .= 2.0
    stokes.viscosity.ηv .= 2.0
    # Converge the separated stress and viscosity updates before differentiating the final state.
    for _ in 1:100
        JustRelax2D.compute_stress_DRYEL!(
            stokes, nonlinear_rheology, phase_ratios, 1.0, 1.0
        )
        JustRelax2D.compute_viscosity_τII!(
            stokes, phase_ratios, nonlinear_args, nonlinear_rheology, (-Inf, Inf);
            relaxation = 0.5,
        )
    end
    nonlinear_scratch = (;
        η_target = copy(stokes.viscosity.η),
        ηv_target = copy(stokes.viscosity.ηv),
        η = similar(stokes.viscosity.η),
        ηv = similar(stokes.viscosity.ηv),
        feedback = similar(stokes.viscosity.η),
        feedback_v = similar(stokes.viscosity.ηv),
        feedback_λ = Ref{Any}(nothing),
        τ = map(similar, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)),
        τ_v = map(similar, (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)),
        τII = similar(stokes.τ.II),
        θ = similar(stokes.ΔPψ),
    )
    # The final-state nonlinear path starts with the stress seed supplied by the momentum
    # residual. Viscosity is a locally eliminated constitutive variable, so a stale viscosity
    # seed must be ignored. The adjoint interface therefore has no relaxation argument.
    adjoint.τ.xx .= 1.0
    adjoint.τ.yy .= -0.25
    adjoint.τ.xy .= 0.5
    adjoint.τ.xy_c .= 0.0
    adjoint.τ.xx_v .= 0.0
    adjoint.τ.yy_v .= 0.0
    adjoint.τ.II .= 0.0
    adjoint.ε.xx .= 0.0
    adjoint.ε.yy .= 0.0
    adjoint.ε.xy .= 0.0
    adjoint.viscosity.η .= 123.0
    adjoint.viscosity.ηv .= 123.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, nonlinear_rheology, phase_ratios, 1.0, 1.0,
        nonlinear_args, (-Inf, Inf), false, nonlinear_scratch,
    )
    @test any(!iszero, adjoint.ε.xx)
    @test any(!iszero, adjoint.ε.xy)
    nonlinear_εxx = copy(adjoint.ε.xx)
    nonlinear_εxy = copy(adjoint.ε.xy)

    adjoint.τ.xx .= 1.0
    adjoint.τ.yy .= -0.25
    adjoint.τ.xy .= 0.5
    adjoint.τ.xy_c .= 0.0
    adjoint.τ.xx_v .= 0.0
    adjoint.τ.yy_v .= 0.0
    adjoint.τ.II .= 0.0
    adjoint.ε.xx .= 0.0
    adjoint.ε.yy .= 0.0
    adjoint.ε.xy .= 0.0
    adjoint.viscosity.η .= -456.0
    adjoint.viscosity.ηv .= -456.0
    JustRelax2D.enzyme_compute_stress_DRYEL!(
        stokes, adjoint, nonlinear_rheology, phase_ratios, 1.0, 1.0,
        nonlinear_args, (-Inf, Inf), false, nonlinear_scratch,
    )
    @test adjoint.ε.xx ≈ nonlinear_εxx
    @test adjoint.ε.xy ≈ nonlinear_εxy

    bcs = VelocityBoundaryConditions()
    adjoint.V.Vx .= 1.0
    adjoint.V.Vy .= 1.0
    JustRelax2D.enzyme_flow_bcs!(stokes, adjoint, bcs)
    @test all(iszero, adjoint.V.Vx[:, 1])
    @test all(iszero, adjoint.V.Vx[:, end])
    @test all(iszero, adjoint.V.Vy[1, :])
    @test all(iszero, adjoint.V.Vy[end, :])

    no_slip = (left = true, right = true, top = true, bot = true)
    bcs = VelocityBoundaryConditions(; no_slip, free_slip = map(!, no_slip))
    adjoint.V.Vx .= 1.0
    adjoint.V.Vy .= 1.0
    JustRelax2D.enzyme_no_slip!(
        stokes.V.Vx, adjoint.V.Vx, stokes.V.Vy, adjoint.V.Vy, bcs.no_slip
    )
    @test all(isfinite, adjoint.V.Vx)
    @test all(isfinite, adjoint.V.Vy)
end
