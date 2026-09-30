using Test, GeoParams, JustRelax

# Load only the diagnostic functions: including the miniapp starts a full simulation.
const source = Meta.parseall(read(joinpath(@__DIR__, "..", "ReservoirFailure2D.jl"), String))
const diagnostic_names = (
    :yield_proximity, :plastic_activity, :pore_pressure_ratio,
    :prescribed_pore_pressure, :failure_diagnostics, :failure_limited_dt, :shear_heating_output, :softened_strength_output, :default_params, :init_rheology, :poisson_ratio,
)
for expr in source.args
    expr isa Expr || continue
    definition = expr.head == :macrocall ? expr.args[end] : expr
    definition isa Expr && definition.head in (:function, :(=)) || continue
    signature = definition.args[1]
    signature isa Expr && signature.head == :call || continue
    signature.args[1] in diagnostic_names && Core.eval(@__MODULE__, expr)
end
const ROCK_PHASE, MAGMA_PHASE, FAULT_PHASE, AIR_PHASE = 1, 2, 3, 4

@testset "Reservoir effective cap diagnostics" begin
    model = DruckerPragerCap(; C = 20.0, ϕ = 30.0, Ψ = 0.0, pT = -1.0)
    cap = (; model, cp = GeoParams.compute_tensile_cap(sind(30.0), cosd(30.0), 0.0, 20.0, -1.0), scale = 1.0)
    pl = DruckerPragerCap(; C = 20.0, ϕ = 30.0, Ψ = 0.0, pT = -1.0)
    for (τ, P) in ((0.0, -2.0), (0.0, -1.0), (5.0, 0.0), (100.0, 100.0))
        r, tensile = yield_proximity(τ, P, cap)
        @test r ≈ max(0, 1 + compute_yieldfunction(pl; P, τII = τ))
    end
    @test yield_proximity(0.0, -1.0, cap) == (1.0, true)
    @test !last(yield_proximity(100.0, 100.0, cap))
    @test plastic_activity(0.0, 0.2, 0.1) == 0.2

    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    nd(x) = nondimensionalize(x, CD)
    p = (;
        λ_mode = :uniform, λ_pf = 1.0, λ_fault = 0.4, α_B = 0.5,
        B = 0.0, Λ_MPa_K = 0.0, halo_width = 3.0, εpl_threshold = 1.0e-6,
        dt_min = 0.01, yield_threshold = 0.8, dt_safety = 0.25,
    )
    phases = reshape([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)], 3, 1)
    pf, ratios = prescribed_pore_pressure(fill(400.0, 3, 1), fill(130.0, 3, 1), phases, ones(3, 1), p)
    @test vec(pf) == [130.0, 52.0, 0.0]
    @test vec(ratios) == [1.0, 0.4, 0.0]

    # Pure volumetric opening with positive total pressure, P - α Pf = -2 MPa.
    stokes = (;
        P = fill(nd(63MPa), 1, 1), τ = (; II = zeros(1, 1)),
        ε_pl = (; II = zeros(1, 1)), ε_vol_pl = fill(0.2, 1, 1), λ = fill(0.1, 1, 1), EII_pl = zeros(1, 1),
    )
    thermal = fill(nd(400C), 3, 3)
    phase_ratios = (; center = fill((1.0, 0.0, 0.0), 1, 1))
    diagnose(st) = failure_diagnostics(
        st, thermal, fill(nd(140MPa), 1, 1), fill(63.0, 1, 1),
        fill(400.0, 1, 1), phase_ratios, (; center = ones(1, 1)), ([1.5], [0.0]), cap, p, CD;
        chamber = (; xc = 0.0, yc = 0.0, r = 1.0), Pf_used = fill(nd(130MPa), 1, 1), λ_used = ones(1, 1)
    )
    d = diagnose(stokes)
    @test d.halo_yield
    @test d.yield_mode == "tensile"
    @test d.r_max_eff > 1
    @test d.r_max < 1
    @test d.tensile_at_max
    shear = diagnose(merge(stokes, (; P = fill(nd(165MPa), 1, 1), τ = (; II = fill(nd(100MPa), 1, 1)))))
    @test shear.halo_yield
    @test shear.yield_mode == "shear" # positive volumetric plasticity alone is not tensile
    @test d.Pf[1] ≈ 130.0 # uses supplied pressure, not recomputed 140 MPa
    inactive = diagnose(merge(stokes, (; ε_vol_pl = zeros(1, 1), λ = zeros(1, 1))))
    @test !inactive.halo_yield
    @test inactive.yield_mode == "none"
    @test isnan(inactive.x_yield)
    @test failure_limited_dt(1.0, 1.0, 0.9, 0.8, p) ≈ 0.25
end


@testset "Shear heating temperature source output" begin
    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    nd(x) = nondimensionalize(x, CD)
    material = SetMaterialParams(;
        Phase = 1, Density = ConstantDensity(; ρ = 2500kg / m^3),
        HeatCapacity = ConstantHeatCapacity(; Cp = 1000J / kg / K), CharDim = CD,
    )
    thermal = (; T = fill(nd(400C), 4, 3), shear_heating = reshape([nd(2J / s / m^3), 0.0], 2, 1))
    saved_T = copy(thermal.T)
    output = shear_heating_output(thermal, zeros(2, 1), ones(Int, 2, 1), (material,), nd(10s), CD)
    @test size(output.dT_shear_source_K) == (2, 1)
    @test output.shear_heating_W_m3[1] ≈ 2.0
    @test output.dT_shear_source_K[1] ≈ 2 * 10 / (2500 * 1000)
    @test output.dT_shear_source_K[2] == 0.0
    @test thermal.T == saved_T
end


@testset "Softened strength VTK fields" begin
    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    soft = DecaySoftening(; εref = 1.0, n = 1.0)
    rock = SetMaterialParams(;
        Phase = 1, CharDim = CD,
        CompositeRheology = CompositeRheology(
            (
                DruckerPragerCap(;
                    C = 20MPa, ϕ = 30.0, pT = -1MPa, softening_C = soft, softening_ϕ = soft,
                ),
            )
        ),
    )
    fault = SetMaterialParams(;
        Phase = 2, CharDim = CD,
        CompositeRheology = CompositeRheology((DruckerPragerCap(; C = 10MPa, ϕ = 15.0, pT = -1MPa),)),
    )
    magma = SetMaterialParams(;
        Phase = 3, CharDim = CD,
        CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e16Pa * s),)),
    )
    strain = reshape([0.0, 1.0, 3.0, 1.0, 1.0], 5, 1)
    phases = reshape(
        [
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.25, 0.75, 0.0), (0.0, 0.0, 1.0),
        ], 5, 1
    )
    output = softened_strength_output(strain, phases, (rock, fault, magma), CD)
    @test size(output.cohesion_MPa) == size(strain)
    @test output.cohesion_MPa[1:4] ≈ [20.0, 10.0, 5.0, 10.0]
    @test output.friction_angle_deg[1:4] ≈ [30.0, 15.0, 7.5, 15.0]
    @test isnan(output.cohesion_MPa[5])
    @test isnan(output.friction_angle_deg[5])
end


@testset "Reservoir crust linear cohesion softening" begin
    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    p = default_params()
    creeps = ntuple(_ -> LinearViscous(; η = 1.0e20Pa * s), 4)
    rheology = init_rheology(creeps, p, CD)
    for (strain, expected) in ((0.0, 20.0), (0.05, 11.0), (0.1, 2.0), (0.2, 2.0))
        _, Ceff, sinϕ, _, _, _ = JustRelax.JustRelax2D.plastic_params(rheology[1], strain)
        @test ustrip(dimensionalize(Ceff, MPa, CD)) ≈ expected
        @test asind(sinϕ) ≈ 30.0
        _, Cf, _, _, _, _ = JustRelax.JustRelax2D.plastic_params(rheology[3], strain)
        @test ustrip(dimensionalize(Cf, MPa, CD)) ≈ 10.0
    end
    model = DruckerPragerCap(;
        C = 20.0, ϕ = 30.0, pT = -10.0,
        softening_C = LinearSoftening(10.0, 20.0, 0.0, 0.1)
    )
    cap = (; model, cp = GeoParams.compute_tensile_cap(sind(30.0), cosd(30.0), 0.0, 20.0, -10.0), scale = 10.0)
    @test first(yield_proximity(12.0, 0.0, cap, 0.1)) > first(yield_proximity(12.0, 0.0, cap, 0.0))
    @test first(yield_proximity(12.0, 0.0, cap, 0.1)) ≈ max(0, 1 + compute_yieldfunction(model; P = 0.0, τII = 12.0, EII = 0.1) / 10)
    @test_throws ArgumentError init_rheology(creeps, merge(p, (; C_softening_end = 0.0)), CD)
end

@testset "Reservoir rheology uses characteristic units" begin
    CD = GEO_units(; length = 14km, viscosity = 1.0e21Pa * s, temperature = 450C)
    p = default_params()
    creeps = ntuple(_ -> LinearViscous(; η = 1.0e20Pa * s), 4)
    rheology = nondimensionalize(init_rheology(creeps, p, CD), CD)
    args = (; T = nondimensionalize(400C, CD), P = nondimensionalize(-130MPa, CD))
    @test JustRelax.JustRelax2D.compute_ρCp(rheology[1], args) > 0
    @test compute_density(rheology[1], args) > 0
    for material in rheology
        @test isfinite(JustRelax.JustRelax2D.compute_ρCp(material, args))
        @test isfinite(GeoParams.compute_conductivity(material, args))
    end
end
