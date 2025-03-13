# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(CharDim; is_plastic = true)

    # Dislocation and Diffusion creep
    disl_upper_crust = DislocationCreep(
        A = 5.07e-18Pa^(-23 // 10) / s, # units are Pa^(-n) / s
        n = 2.3NoUnits,
        E = 154.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    disl_lower_crust = DislocationCreep(
        A = 2.08e-23Pa^(-32 // 10) / s, # units are Pa^(-n) / s
        n = 3.2NoUnits,
        E = 238.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )
    disl_lithospheric_mantle = DislocationCreep(
        A = 2.51e-17Pa^(-35 // 10) / s, # units are Pa^(-n) / s
        n = 3.5NoUnits,
        E = 530.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    disl_sublithospheric_mantle = DislocationCreep(
        A = 2.51e-17Pa^(-35 // 10) / s, # units are Pa^(-n) / s
        n = 3.5NoUnits,
        E = 530.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    diff_lithospheric_mantle = DiffusionCreep(
        A = 2.51e-17Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
        n = 1.0NoUnits,
        E = 530.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        p = 0.0NoUnits,
        R = 8.3145J / mol / K
    )
    diff_sublithospheric_mantle = DiffusionCreep(
        A = 2.51e-17Pa^(-1) / s, # units are Pa^(-n - r) / s * m^(-p)
        n = 1.0NoUnits,
        E = 530.0e3J / mol,
        V = 6.0e-6m^3 / mol,
        r = 0.0NoUnits,
        p = 0.0NoUnits,
        R = 8.3145J / mol / K
    )

    # Elasticity
    el_upper_crust = SetConstantElasticity(; G = 25.0e9Pa, ν = 0.5)
    el_lower_crust = SetConstantElasticity(; G = 25.0e9Pa, ν = 0.5)
    el_lithospheric_mantle = SetConstantElasticity(; G = 67.0e9Pa, ν = 0.5)
    el_sublithospheric_mantle = SetConstantElasticity(; G = 67.0e9Pa, ν = 0.5)
    β_upper_crust = inv(get_Kb(el_upper_crust))
    β_lower_crust = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg = 1.0e16 * Pa * s
    cohesion = 3.0e6 * Pa
    friction = asind(0.2)
    pl_crust = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    end
    friction = asind(0.3)
    pl = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
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
            Density = PT_Density(; ρ0 = 2.75e3kg / m^3, β = β_upper_crust, T0 = 0.0e0C, α = 3.5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity = el_upper_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            Gravity = ConstantGravity(; g = g),
            CharDim = CharDim,
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.0e3kg / m^3, β = β_upper_crust, T0 = 0.0e0C, α = 3.5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lower_crust,
            CharDim = CharDim,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_upper_crust, T0 = 0.0e0C, α = 3.5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lithospheric_mantle,
            CharDim = CharDim,
        ),
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = (3.3e3 - 50)kg / m^3, β = β_upper_crust, T0 = 0.0e0C, α = 3.5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
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
            Density = ConstantDensity(; ρ = 1.0e3kg / m^3), # water density
            HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3J / kg / K),
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            Conductivity = ConstantConductivity(; k = 1.0Watt / m / K),
            Gravity = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19Pa * s),)),
            CharDim = CharDim,
        ),
    )
end

function init_phases!(phases, particles, Lx, d, r, thick_air, CharDim)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx, CharDim)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j]) - nondimensionalize(thick_air * km, CharDim)
            if nondimensionalize(0.0e0km, CharDim) ≤ depth ≤ nondimensionalize(21km, CharDim)
                @index phases[ip, i, j] = 1.0

            elseif nondimensionalize(35km, CharDim) ≥ depth > nondimensionalize(21km, CharDim)
                @index phases[ip, i, j] = 2.0

            elseif nondimensionalize(90km, CharDim) ≥ depth > nondimensionalize(35km, CharDim)
                @index phases[ip, i, j] = 3.0

            elseif depth > nondimensionalize(90km, CharDim)
                @index phases[ip, i, j] = 3.0

            elseif depth < nondimensionalize(0.0e0km, CharDim)
                @index phases[ip, i, j] = 5.0

            end

            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && (((@index py[ip, i, j]) - d - nondimensionalize(thick_air * km, CharDim))^2 ≤ r^2)
                @index phases[ip, i, j] = 4.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, CharDim)
end
