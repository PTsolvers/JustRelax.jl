function init_rheologies(CharDim; is_plastic = true)

    # Dislocation and Diffusion creep
    # disl_upper_crust = DislocationCreep(
    #     A = 5.07e-18Pa^(-23 // 10) / s, # units are Pa^(-n) / s
    #     n = 2.3NoUnits,
    #     E = 154.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # disl_lower_crust = DislocationCreep(
    #     A = 2.08e-23Pa^(-32 // 10) / s, # units are Pa^(-n) / s
    #     n = 3.2NoUnits,
    #     E = 238.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K,
    # )
    # disl_lithospheric_mantle = DislocationCreep(
    #     A = 2.51e-17Pa^(-35 // 10) / s, # units are Pa^(-n) / s
    #     n = 3.5NoUnits,
    #     E = 530.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # disl_sublithospheric_mantle = DislocationCreep(
    #     A = 2.51e-17Pa^(-35 // 10) / s, # units are Pa^(-n) / s
    #     n = 3.5NoUnits,
    #     E = 530.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # diff_lithospheric_mantle = DiffusionCreep(
    #     A = 2.51e-17Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
    #     n = 1.0NoUnits,
    #     E = 530.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     p = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # diff_sublithospheric_mantle = DiffusionCreep(
    #     A = 2.51e-17Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
    #     n = 1.0NoUnits,
    #     E = 530.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     p = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )

    # disl_upper_crust            = SetDislocationCreep(GeoParams.Dislocation.wet_quartzite_Hirth_2001)
    # disl_lower_crust            = SetDislocationCreep(GeoParams.Dislocation.plagioclase_An75_Ji_1993)
    # disl_lithospheric_mantle    = SetDislocationCreep(GeoParams.Dislocation.wet_olivine_Hirth_2003)
    # disl_sublithospheric_mantle = SetDislocationCreep(GeoParams.Dislocation.dry_olivine_Hirth_2003)
    # diff_lithospheric_mantle    = SetDiffusionCreep(GeoParams.Diffusion.dry_olivine_Hirth_2003)
    # diff_sublithospheric_mantle = SetDiffusionCreep(GeoParams.Diffusion.wet_olivine_Hirth_2003)
    
    # disl_upper_crust = DislocationCreep(
    #     A = 1.97e-17Pa^(-23//10) / s, # units are Pa^(-n) / s
    #     n = 2.3NoUnits,
    #     E = 153.0e3J / mol,
    #     V = 0.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # disl_lower_crust = DislocationCreep(
    #     A = 1.26e-17Pa^(-34//10) / s, # units are Pa^(-n) / s
    #     n = 3.4NoUnits,
    #     E = 260.0e3J / mol,
    #     V = 0.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K,
    # )
    # disl_lithospheric_mantle = DislocationCreep(
    #     A = 1.1e-16Pa^(-35 // 10) / s, # units are Pa^(-n) / s
    #     n = 3.5NoUnits,
    #     E = 530.0e3J / mol,
    #     V = 13.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # disl_sublithospheric_mantle = DislocationCreep(
    #     A = 5.01e-20Pa^(-4) / s, # units are Pa^(-n) / s
    #     n = 4.0NoUnits,
    #     E = 470.0e3J / mol,
    #     V = 10.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # diff_lithospheric_mantle = DiffusionCreep(
    #     A = (10^-8.16)Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
    #     n = 1.0NoUnits,
    #     E = 375.0e3J / mol,
    #     V = 6.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     p = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )
    # diff_sublithospheric_mantle = DiffusionCreep(
    #     A = (10^-8.64)Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
    #     n = 1.0NoUnits,
    #     E = 350.0e3J / mol,
    #     V = 4.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     p = 0.0NoUnits,
    #     R = 8.3145J / mol / K
    # )

    # From Naliboff
    disl_upper_crust = DislocationCreep(
        A = 8.57e-28Pa^(-4) / s, # units are Pa^(-n) / s
        n = 4.0NoUnits,
        E = 223.0e3J / mol,
        V = 0.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    disl_middle_crust = DislocationCreep(
        A = 1.26e-17Pa^(-34//10) / s, # units are Pa^(-n) / s
        n = 3.4NoUnits,
        E = 223.0e3J / mol,
        V = 0.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )
    disl_lower_crust = DislocationCreep(
        A = 7.13e-18Pa^(-3) / s, # units are Pa^(-n) / s
        n = 3.0NoUnits,
        E = 345.0e3J / mol,
        V = 0.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )
    disl_lithospheric_mantle = DislocationCreep(
        A = 6.52e-16Pa^(-35 // 10) / s, # units are Pa^(-n) / s
        n = 3.5NoUnits,
        E = 530.0e3J / mol,
        V = 18.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    disl_sublithospheric_mantle = DislocationCreep(
        A = 6.52e-16Pa^(-35//10) / s, # units are Pa^(-n) / s
        n = 3.5NoUnits,
        E = 530.0e3J / mol,
        V = 18.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    diff_lithospheric_mantle = DiffusionCreep(
        A = (10^-8.16)Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
        n = 1.0NoUnits,
        E = 375.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        p = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    diff_sublithospheric_mantle = DiffusionCreep(
        A = (2.37^-15)Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
        n = 1.0NoUnits,
        E = 375.0e3J / mol,
        V = 10.0e-6m^3 / mol,
        r = 0.0NoUnits,
        p = 3.0NoUnits,
        d = 5.3e-3m,
        R = 8.3145J / mol / K
    )

    # Elasticity
    el_upper_crust            = SetConstantElasticity(; G = 36.0e9Pa, ν = 0.4)
    el_lower_crust            = SetConstantElasticity(; G = 40.0e9Pa, ν = 0.4)
    el_lithospheric_mantle    = SetConstantElasticity(; G = 74.0e9Pa, ν = 0.4)
    el_sublithospheric_mantle = SetConstantElasticity(; G = 74.0e9Pa, ν = 0.4)
    β_upper_crust             = inv(get_Kb(el_upper_crust))
    β_lower_crust             = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle     = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle  = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    # η_reg    = 1e17 * Pa * s
    η_reg    = 2.0e20 * Pa * s
    cohesion = 40e6 * Pa
    friction = 30NoUnits

    # soft_C  = LinearSoftening((cohesion/10, cohesion), (0e0, 1e0))
    soft_C = NonLinearSoftening(; ξ₀ = cohesion, Δ = cohesion * 0.9)

    pl_crust = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    end
    # friction = asind(0.3)
    pl = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = 30NoUnits, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = 30NoUnits, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    end

    K_crust = TP_Conductivity(;
        a = 0.64Watt / K / m,
        b = 807.0e0Watt / m,
        c = 0.77K,
        d = 0.00004 / MPa,
    )
    K_mantle = TP_Conductivity(;
        a = 0.73Watt / K / m,
        b = 1293.0e0Watt / m,
        c = 0.77K,
        d = 0.00004 / MPa,
    )

    g = 9.81m / s^2

    # Define rheolgy struct
    return rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.75e3kg / m^3, β = β_upper_crust, T0 = 273e0K, α = 2.4e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity = el_upper_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.97e-6Watt/m^3),
            Gravity = ConstantGravity(; g = g),
            CharDim = CharDim,
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.85e3kg / m^3, β = β_upper_crust, T0 = 273e0K, α = 2.4e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.97e-6Watt/m^3),
            # CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lower_crust,
            CharDim = CharDim,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_upper_crust, T0 = 273e0K, α = 3e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lithospheric_mantle,
            CharDim = CharDim,
        ),
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_upper_crust, T0 = 273e0K, α = 3e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_sublithospheric_mantle,
            CharDim = CharDim,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 1.0kg / m^3), # water density
            HeatCapacity = ConstantHeatCapacity(; Cp = 1e5J / kg / K),
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            Conductivity = ConstantConductivity(; k = 1.0Watt / m / K),
            Gravity = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(;η=1e18Pa*s), )),
            Elasticity = el_upper_crust,
            CharDim = CharDim,
        ),
    )
end

function init_phases!(phases, particles, CharDim)
    ni = size(phases)

    d0  = nondimensionalize(0km, CharDim)
    d20 = nondimensionalize(22km, CharDim)
    d40 = nondimensionalize(40km, CharDim)
    d90 = nondimensionalize(100km, CharDim)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j]) # - nondimensionalize(thick_air * km, CharDim)
            if d0 ≤ depth ≤ d20
                @index phases[ip, i, j] = 1.0

            elseif d40 ≥ depth > d20
                @index phases[ip, i, j] = 2.0

            elseif d90 ≥ depth > d40
                @index phases[ip, i, j] = 3.0

            elseif depth > d90
                @index phases[ip, i, j] = 4.0

            elseif depth < d0
                @index phases[ip, i, j] = 5.0

            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
