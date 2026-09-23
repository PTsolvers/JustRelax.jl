using Test
using GeoParams
using Enzyme
using JustRelax, JustRelax.JustRelax2D

function _local_material_objective(material, args)
    density = compute_density(material, args)
    return density + JustRelax2D.get_shear_modulus(material)
end

function _plastic_material(parameters)
    elasticity = ConstantElasticity(; G = 1.5, Kb = 4.0)
    plasticity = DruckerPrager_regularised(; parameters...)
    return SetMaterialParams(;
        Phase = 1,
        CompositeRheology = CompositeRheology((
            LinearViscous(; η = 2.0), elasticity, plasticity,
        )),
        Elasticity = elasticity,
        Plasticity = plasticity,
    )
end

@testset "Plastic parameter extraction from local stress gradient" begin
    parameters = (; C = 0.15, ϕ = 25.0, Ψ = 8.0, η_vp = 0.07)
    material = _plastic_material(parameters)
    args = (
        (0.8, -0.4, 0.5),
        (0.1, -0.05, 0.03),
        2.0,
        0.2,
        0.0,
        1.0,
        0.5,
        0.0,
        (0.7, -0.2, 0.4),
        0.6,
        0.8,
    )
    solution = JustRelax2D._compute_local_stress(
        args[1], args[2], args[3], args[4],
        GeoParams.get_G(material), GeoParams.get_Kb(material),
        args[5], args[6], material, args[7], args[8],
    )
    @test solution[8] > 0

    derivative, dη = JustRelax2D.enzyme_stress_gradients(material, args...)
    hη = 1.0e-6
    plusη = Base.setindex(args, args[3] + hη, 3)
    minusη = Base.setindex(args, args[3] - hη, 3)
    finite_difference_η = (
        JustRelax2D.stress_parameter_objective(material, plusη...) -
            JustRelax2D.stress_parameter_objective(material, minusη...)
    ) / (2hη)
    @test dη ≈ finite_difference_η rtol = 1.0e-6 atol = 1.0e-8
    for name in keys(parameters)
        path = (:CompositeRheology, 1, :elements, 3, name)
        gradient = JustRelax2D.material_parameter_gradient(material, derivative, Val(path))
        h = name in (:ϕ, :Ψ) ? 1.0e-5 : 1.0e-6
        plus = merge(parameters, NamedTuple{(name,)}((parameters[name] + h,)))
        minus = merge(parameters, NamedTuple{(name,)}((parameters[name] - h,)))
        finite_difference = (
            JustRelax2D.stress_parameter_objective(_plastic_material(plus), args...) -
                JustRelax2D.stress_parameter_objective(_plastic_material(minus), args...)
        ) / (2h)
        @test gradient ≈ finite_difference rtol = 1.0e-6 atol = 1.0e-8
    end
end

@testset "General local material-object gradient" begin
    args = (; T = 3.0, P = 2.0)
    density = PT_Density(; ρ0 = 2.0, α = 0.1, β = 0.2, T0 = 1.0, P0 = 0.5)
    elasticity = ConstantElasticity(; G = 2.0, Kb = 10.0)
    material = SetMaterialParams(;
        Phase = 1,
        Density = density,
        Elasticity = elasticity,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 3.0), elasticity)),
    )
    baseline = _local_material_objective(material, args)

    derivative = JustRelax2D.enzyme_material_gradient(
        _local_material_objective, material, args
    )

    @test JustRelax2D.material_parameter_gradient(
        material, derivative, Val((:Density, 1, :α))
    ) ≈ -4.0
    @test JustRelax2D.material_parameter_gradient(
        material, derivative, Val((:Density, 1, :ρ0))
    ) ≈ 1.1
    @test JustRelax2D.material_parameter_gradient(
        material, derivative, Val((:CompositeRheology, 1, :elements, 2, :G))
    ) ≈ 1.0
    # GeoParams also stores the supplied elasticity separately, but this particular
    # objective reads G through CompositeRheology. The path therefore identifies the
    # actual forward input instead of merging same-named fields implicitly.
    @test iszero(
        JustRelax2D.material_parameter_gradient(
            material, derivative, Val((:Elasticity, 1, :G))
        )
    )
    @test _local_material_objective(material, args) == baseline
end

# Run with: julia --project=. --startup-file=no test/test_enzyme_geoparams.jl
# Differentiate the unchanged GeoParams function with respect to the immutable model.
# The returned model-shaped derivative stores numeric derivatives in each field's .val;
# its unit metadata should not be interpreted as units of the derivative.
@testset "Direct GeoParams PT_Density parameter gradients" begin
    args = (; T = 3.0, P = 2.0)
    for α in (0.1, 0.0)
        parameters = (; ρ0 = 2.0, α, β = 0.2, T0 = 1.0, P0 = 0.5)
        density = PT_Density(; parameters...)
        baseline = compute_density(density, args)

        derivatives = Enzyme.autodiff(
            Enzyme.Reverse,
            compute_density,
            Enzyme.Active,
            Enzyme.Active(density),
            Enzyme.Const(args),
        )[1][1]

        expected = (;
            ρ0 = 1 - α * (args.T - parameters.T0) + parameters.β * (args.P - parameters.P0),
            α = -parameters.ρ0 * (args.T - parameters.T0),
            β = parameters.ρ0 * (args.P - parameters.P0),
            T0 = parameters.ρ0 * α,
            P0 = -parameters.ρ0 * parameters.β,
        )
        @test typeof(derivatives) === typeof(density)
        for name in keys(parameters)
            gradient = getproperty(derivatives, name).val
            @test gradient ≈ expected[name] atol = 1.0e-12

            # Reconstruct perturbed models only for the independent finite-difference check.
            h = 1.0e-6
            plus = merge(parameters, NamedTuple{(name,)}((parameters[name] + h,)))
            minus = merge(parameters, NamedTuple{(name,)}((parameters[name] - h,)))
            finite_difference = (
                compute_density(PT_Density(; plus...), args) -
                    compute_density(PT_Density(; minus...), args)
            ) / (2h)
            @test gradient ≈ finite_difference rtol = 1.0e-8 atol = 1.0e-9
            @test getproperty(density, name).val == parameters[name]
        end
        @test compute_density(density, args) == baseline
    end
end
