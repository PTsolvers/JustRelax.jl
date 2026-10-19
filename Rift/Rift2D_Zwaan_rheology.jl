# Material properties of the reference models of Zwaan et al. (2026, JGR Solid Earth,
# doi:10.1029/2025JB033255), Table A1. Phases: 1 upper crust, 2 lower crust,
# 3 lithospheric mantle, 4 sub-lithospheric mantle, 5 sticky air.
#
# The flow-law prefactors in Table A1 follow ASPECT's convention, ε̇II = A τII^n d^-m
# exp(-(E + PV) / RT) with second invariants, which is GeoParams' `Invariant` apparatus.
# Thermal conductivities are k = κ ρ0 Cp from the tabulated diffusivities.

# Density at T0 = 273 K and conductivity k = κ ρ0 Cp, from Table A1
const ZWAAN_ρ0 = (upper_crust = 2700.0, lower_crust = 2850.0, lith_mantle = 3280.0, asthenosphere = 3300.0)
const ZWAAN_k = (
    upper_crust = 7.72e-7 * 2700 * 1200,
    lower_crust = 7.31e-7 * 2850 * 1200,
    lith_mantle = 8.38e-7 * 3280 * 1200,
    asthenosphere = 8.33e-7 * 3300 * 1200,
)
const ZWAAN_H = (upper_crust = 1.0e-6, lower_crust = 0.1e-6, lith_mantle = 0.0, asthenosphere = 0.0)

function init_rheologies(CharDim; η_vp = 1.0e20 * Pa * s, pT_factor = 0.5, tension_cap = true)

    R = 8.3145J / mol / K
    d = 1.0e-3m # grain size

    # Dislocation and diffusion creep
    disl_quartzite = DislocationCreep(;
        A = 8.57e-28Pa^(-4) / s, n = 4.0NoUnits, E = 223.0e3J / mol, V = 0.0m^3 / mol,
        r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    diff_quartzite = DiffusionCreep(;
        A = 5.79e-19Pa^(-1) / s * m^2, n = 1.0NoUnits, E = 223.0e3J / mol, V = 0.0m^3 / mol,
        d = d, p = -2.0NoUnits, r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    disl_anorthite = DislocationCreep(;
        A = 7.13e-18Pa^(-3) / s, n = 3.0NoUnits, E = 345.0e3J / mol, V = 38.0e-6m^3 / mol,
        r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    diff_anorthite = DiffusionCreep(;
        A = 2.99e-25Pa^(-1) / s * m^3, n = 1.0NoUnits, E = 159.0e3J / mol, V = 38.0e-6m^3 / mol,
        d = d, p = -3.0NoUnits, r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    disl_dry_olivine = DislocationCreep(;
        A = 6.52e-16Pa^(-7 // 2) / s, n = 3.5NoUnits, E = 530.0e3J / mol, V = 18.0e-6m^3 / mol,
        r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    diff_dry_olivine = DiffusionCreep(;
        A = 2.25e-9Pa^(-1) / s, n = 1.0NoUnits, E = 375.0e3J / mol, V = 6.0e-6m^3 / mol,
        d = d, p = 0.0NoUnits, r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    disl_wet_olivine = DislocationCreep(;
        A = 2.12e-15Pa^(-7 // 2) / s, n = 3.5NoUnits, E = 480.0e3J / mol, V = 11.0e-6m^3 / mol,
        r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )
    diff_wet_olivine = DiffusionCreep(;
        A = 1.5e-9Pa^(-1) / s, n = 1.0NoUnits, E = 335.0e3J / mol, V = 4.0e-6m^3 / mol,
        d = d, p = 0.0NoUnits, r = 0.0NoUnits, R = R, Apparatus = Invariant,
    )

    # Elasticity: not part of the visco-plastic models of Zwaan et al.; the shear moduli
    # make the Maxwell time short compared with the tectonic time scale.
    el_crust = SetConstantElasticity(; G = 36.0e9Pa, ν = 0.3)
    el_mantle = SetConstantElasticity(; G = 74.0e9Pa, ν = 0.3)

    # Drucker-Prager with a tensile cap at pT = -pT_factor·C (default -C/2). The friction angle drops linearly
    # by a factor of 4 (26.56° to 6.64°) as the plastic strain goes from 0 to 1.
    cohesion = 5.0e6Pa
    friction = 26.56NoUnits
    soft_ϕ = LinearSoftening(friction / 4, friction, 0.0NoUnits, 1.0NoUnits)
    # `tension_cap = false` drops the cap (plain regularized Drucker-Prager).
    # pl = if tension_cap
    #     DruckerPragerCap(;
    #         C = cohesion, ϕ = friction, Ψ = 0.0NoUnits, η_vp = η_vp, pT = -pT_factor * cohesion,
    #         softening_ϕ = soft_ϕ,
    #     )
    # else
    #     DruckerPrager_regularised(;
    #         C = cohesion, ϕ = friction, Ψ = 0.0NoUnits, η_vp = η_vp, softening_ϕ = soft_ϕ,
    #     )
    # end

    pl = DruckerPragerCap(;
            C = cohesion, ϕ = friction, Ψ = 10.0NoUnits, η_vp = η_vp, pT = -pT_factor * cohesion,
            softening_ϕ = soft_ϕ,
        )
    # pl = DruckerPrager_regularised(;
    #     C = cohesion, ϕ = friction, Ψ = 10.0NoUnits, η_vp = η_vp, softening_ϕ = soft_ϕ,
    # )

    g = ConstantGravity(; g = 9.81m / s^2)
    Cp = ConstantHeatCapacity(; Cp = 1200.0J / kg / K)

    return (
        # Name = "UpperCrust",
        SetMaterialParams(;
            Phase = 1,
            Density = T_Density(; ρ0 = ZWAAN_ρ0.upper_crust * kg / m^3, α = 2.7e-5 / K, T0 = 273.0K),
            HeatCapacity = Cp,
            Conductivity = ConstantConductivity(; k = ZWAAN_k.upper_crust * Watt / m / K),
            RadioactiveHeat = ConstantRadioactiveHeat(ZWAAN_H.upper_crust * Watt / m^3),
            CompositeRheology = CompositeRheology((disl_quartzite, diff_quartzite, el_crust, pl)),
            Elasticity = el_crust,
            Gravity = g,
            CharDim = CharDim,
        ),
        # Name = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = T_Density(; ρ0 = ZWAAN_ρ0.lower_crust * kg / m^3, α = 2.7e-5 / K, T0 = 273.0K),
            HeatCapacity = Cp,
            Conductivity = ConstantConductivity(; k = ZWAAN_k.lower_crust * Watt / m / K),
            RadioactiveHeat = ConstantRadioactiveHeat(ZWAAN_H.lower_crust * Watt / m^3),
            CompositeRheology = CompositeRheology((disl_anorthite, diff_anorthite, el_crust, pl)),
            Elasticity = el_crust,
            Gravity = g,
            CharDim = CharDim,
        ),
        # Name = "LithosphericMantle",
        SetMaterialParams(;
            Phase = 3,
            Density = T_Density(; ρ0 = ZWAAN_ρ0.lith_mantle * kg / m^3, α = 3.0e-5 / K, T0 = 273.0K),
            HeatCapacity = Cp,
            Conductivity = ConstantConductivity(; k = ZWAAN_k.lith_mantle * Watt / m / K),
            RadioactiveHeat = ConstantRadioactiveHeat(ZWAAN_H.lith_mantle * Watt / m^3),
            CompositeRheology = CompositeRheology((disl_dry_olivine, diff_dry_olivine, el_mantle, pl)),
            Elasticity = el_mantle,
            Gravity = g,
            CharDim = CharDim,
        ),
        # Name = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase = 4,
            Density = T_Density(; ρ0 = ZWAAN_ρ0.asthenosphere * kg / m^3, α = 3.0e-5 / K, T0 = 273.0K),
            HeatCapacity = Cp,
            Conductivity = ConstantConductivity(; k = ZWAAN_k.asthenosphere * Watt / m / K),
            RadioactiveHeat = ConstantRadioactiveHeat(ZWAAN_H.asthenosphere * Watt / m^3),
            CompositeRheology = CompositeRheology((disl_wet_olivine, diff_wet_olivine, el_mantle, pl)),
            Elasticity = el_mantle,
            Gravity = g,
            CharDim = CharDim,
        ),
        # Name = "StickyAir",
        # The huge heat capacity freezes the air at the surface temperature it is reset to
        # every step, so the rock surface sees a fixed 0 °C as in Zwaan et al.
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 1.0kg / m^3),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.0e8J / kg / K),
            Conductivity = ConstantConductivity(; k = 3.0Watt / m / K),
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19Pa * s),)),
            Gravity = g,
            CharDim = CharDim,
        ),
    )
end
