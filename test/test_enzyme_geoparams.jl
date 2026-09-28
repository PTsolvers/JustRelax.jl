using Test
using GeoParams
using Enzyme
using JustRelax, JustRelax.JustRelax2D

function _local_material_objective(material, args)
    density = compute_density(material, args)
    return density + JustRelax2D.get_shear_modulus(material)
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
