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
    # down to 660km
    A= 1.5e8
    A= 2e-23
    # disl_upper_mantle = DislocationCreep( # dry olivine
    #     # A = 2.08e-23Pa^(-35 // 10) / s, # units are Pa^(-n) / s
    #     # A = (10^9.2)MPa^(-35 // 10) / s,   # material specific rheological parameter
    #     A = (A)MPa^(-35 // 10) / s,   # material specific rheological parameter
    #     n = 3.5NoUnits,
    #     E = 532.0e3J / mol,
    #     V = 14.0e-6m^3 / mol,
    #     r = 0.0NoUnits,
    #     R = 8.3145J / mol / K,
    # )
    disl_upper_mantle = DislocationCreep( # dry olivine
        A = (A)MPa^(-35 // 10) / s,   # material specific rheological parameter
        n = 3.5NoUnits,
        E = 230.0e3J / mol,
        V = 11.0e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )
    # down to 2600km
    disl_pv_mantle = DislocationCreep(
        # A = 2.08e-23Pa^(-35 // 10) / s, # units are Pa^(-n) / s
        # A = (10^9.2)MPa^(-35 // 10) / s,   # material specific rheological parameter
        A = (A)MPa^(-35 // 10) / s,   # material specific rheological parameter
        n = 3.5NoUnits,
        E = 370.0e3J / mol,
        V = 3.65e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )
    # down to CMB
    disl_ppv_mantle = DislocationCreep(
        # A = 2.08e-23Pa^(-35 // 10) / s, # units are Pa^(-n) / s
        # A = (10^9.2)MPa^(-35 // 10) / s,   # material specific rheological parameter
        A = (A)MPa^(-35 // 10) / s,   # material specific rheological parameter
        n = 3.5NoUnits,
        E = 162.0e3J / mol,
        V = 1.4e-6m^3 / mol,
        r = 0.0NoUnits,
        R = 8.3145J / mol / K,
    )


    A= 10^9.2
    # down to 660km
    diff_upper_mantle = DiffusionCreep( # dry olivine
        n = 1.0NoUnits,                         # power-law exponent
        r = 0.0NoUnits,                         # exponent of water-fugacity
        p = -2.0NoUnits,                        # grain size exponent
        A = (A)MPa^(-1) * μm^3 * s^(-1),        # material specific rheological parameter
        E = 125.0kJ / mol,                      # activation energy
        V = 7.0e-6m^3 / mol,                    # activation Volume
    )
    # diff_upper_mantle = DiffusionCreep( # dry olivine
    #     n = 1.0NoUnits,                         # power-law exponent
    #     r = 0.0NoUnits,                         # exponent of water-fugacity
    #     p = -3.0NoUnits,                        # grain size exponent
    #     A = (A)MPa^(-1) * μm^3 * s^(-1),        # material specific rheological parameter
    #     E = 300.0kJ / mol,                      # activation energy
    #     V = 5.0e-6m^3 / mol,                    # activation Volume
    # )
    # down to 2600km
    diff_pv_mantle = DiffusionCreep(
        n = 1.0NoUnits,                         # power-law exponent
        r = 0.0NoUnits,                         # exponent of water-fugacity
        p = -3.0NoUnits,                        # grain size exponent
        A = (A)MPa^(-1) * μm^3 * s^(-1),        # material specific rheological parameter
        E = 370.0kJ / mol,                      # activation energy
        V = 3.65e-6m^3 / mol,                   # activation Volume
    )
    # down to CMB
    diff_ppv_mantle = DiffusionCreep(
        n = 1.0NoUnits,                         # power-law exponent
        r = 0.0NoUnits,                         # exponent of water-fugacity
        p = -3.0NoUnits,                        # grain size exponent
        A = (A)MPa^(-1) * μm^3 * s^(-1),   # material specific rheological parameter
        E = 162.0kJ / mol,                      # activation energy
        V = 1.4e-6m^3 / mol,                    # activation Volume
    )

    # Elasticity
    el_upper_crust = SetConstantElasticity(;  G = 25.0e9Pa, ν = 0.4)
    el_lower_crust = SetConstantElasticity(;  G = 25.0e9Pa, ν = 0.4)
    el_upper_mantle = SetConstantElasticity(; G = 60.0e9Pa, ν = 0.4)
    el_lower_mantle = SetConstantElasticity(; G = 60.0e9Pa, ν = 0.4)
    β_upper_crust = inv(get_Kb(el_upper_crust))
    β_lower_crust = inv(get_Kb(el_lower_crust))
    β_upper_mantle = inv(get_Kb(el_upper_mantle))
    β_lower_mantle = inv(get_Kb(el_lower_mantle))

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
            Density = PT_Density(; ρ0 = 2.75e3kg / m^3, β = β_upper_crust, T0 = 0.0e0C, α = 3e-5 / K),
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
            Density = PT_Density(; ρ0 = 2.85e3kg / m^3, β = β_lower_crust, T0 = 0.0e0C, α = 3e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2J / kg / K),
            Conductivity = K_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lower_crust,
            CharDim = CharDim,
        ),
        # Name              = "Down to 660",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_upper_mantle, T0 = 0.0e0C, α = 5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_upper_mantle, diff_upper_mantle, el_upper_mantle, pl)),
            # CompositeRheology = CompositeRheology((diff_upper_mantle, el_upper_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12Watt / kg),
            Elasticity = el_upper_mantle,
            CharDim = CharDim,
        ),
        # Name              = "Down to 2700"
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_upper_mantle, T0 = 0.0e0C, α = 5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_pv_mantle, diff_pv_mantle, el_upper_mantle, pl)),
            # CompositeRheology = CompositeRheology(( diff_pv_mantle, el_upper_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12Watt / kg),
            Elasticity = el_upper_mantle,
            CharDim = CharDim,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 5,
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, β = β_lower_mantle, T0 = 0.0e0C, α = 5e-5 / K),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.25e3J / kg / K),
            Conductivity = K_mantle,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_ppv_mantle, diff_ppv_mantle, el_lower_mantle, pl)),
            # CompositeRheology = CompositeRheology(( diff_ppv_mantle, el_lower_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12Watt / kg),
            Elasticity = el_lower_mantle,
            CharDim = CharDim,
        ),
    )
end

function init_phases!(phases, particles, Lx, d, r, thick_air, CharDim)
    ni = size(phases)

    d_air = nondimensionalize(thick_air * km, CharDim)
    d_0km = nondimensionalize(0.0e0km, CharDim)
    d_21km = nondimensionalize(21.0e0km, CharDim)
    d_35km = nondimensionalize(35.0e0km, CharDim)
    d_660km = nondimensionalize(660.0e0km, CharDim)
    d_2700km = nondimensionalize(2790.0e0km, CharDim)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            depth = -(@index py[ip, i, j]) - d_air
            if d_0km ≤ depth ≤ d_21km
                @index phases[ip, i, j] = 1.0

            elseif d_35km ≥ depth > d_21km
                @index phases[ip, i, j] = 2.0

            elseif d_660km ≥ depth > d_35km
                @index phases[ip, i, j] = 3.0

            elseif d_2700km > depth > d_660km
                @index phases[ip, i, j] = 4.0

            else#if depth < d_2700km
                @index phases[ip, i, j] = 5.0

            end

            # # plume - rectangular
            # if ((x - Lx * 0.5)^2 ≤ r^2) && (((@index py[ip, i, j]) - d - d_air)^2 ≤ r^2)
            #     @index phases[ip, i, j] = 4.0
            # end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx)
end


# Smooth Earth-like double-boundary-layer temperature profile
# with an adiabatic interior, plus optional 2D sinusoidal perturbation.

# Coordinates:
#   z = 0 at top, z = Lz at bottom
# Units in this example:
#   km for x,z,Lx,Lz,delta_t,delta_b,w_t,w_b,zm
#   K for temperatures
#   K/km for gamma

# Smooth switch
S(ξ) = 0.5 * (1.0 + tanh(ξ))

# Interior adiabat
function T_ad(z; Tm=1600.0, zm=100.0, gamma=0.35)
    return Tm + gamma * (z - zm)
end

# Smooth 1D background profile
function T_bg(
    z;
    Lz=2890.0,
    Ttop=273.0,
    Tbot=3800.0,
    Tm=1600.0,
    zm=100.0,
    gamma=0.35,
    delta_t=120.0,
    delta_b=150.0,
    w_t=20.0,
    w_b=25.0,
)
    # Smooth switches
    s_t = S((z - delta_t) / w_t)
    s_b = S(((Lz - delta_b) - z) / w_b)

    # Matching temperatures on adiabat
    Tad_topmatch = T_ad(delta_t; Tm=Tm, zm=zm, gamma=gamma)
    Tad_botmatch = T_ad(Lz - delta_b; Tm=Tm, zm=zm, gamma=gamma)

    # Top conductive branch
    T_topBL = Ttop + (Tad_topmatch - Ttop) * (z / delta_t)

    # Bottom conductive branch
    T_botBL = Tad_botmatch + (Tbot - Tad_botmatch) * ((z - (Lz - delta_b)) / delta_b)

    # Smoothly blended profile
    return (1.0 - s_t) * T_topBL + (s_t * s_b) * T_ad(z; Tm=Tm, zm=zm, gamma=gamma) + (1.0 - s_b) * T_botBL
end

# Full 2D temperature field
function T_field(
    x, z;
    Lx=5780.0,
    Lz=2890.0,
    A=5.0,      # perturbation amplitude in K
    n=1,
    m=1,
    Ttop=273.0,
    Tbot=3800.0,
    Tm=1600.0 + 200,
    zm=100.0,
    gamma=0.35,
    delta_t=110.0,
    delta_b=100.0,
    w_t=20.0,
    w_b=25.0,
)
    background = T_bg(
        z;
        Lz=Lz,
        Ttop=Ttop,
        Tbot=Tbot,
        Tm=Tm,
        zm=zm,
        gamma=gamma,
        delta_t=delta_t,
        delta_b=delta_b,
        w_t=w_t,
        w_b=w_b,
    )

    # perturbation = A * sin(n * π * x / Lx) * sin(m * π * z / Lz)
    perturbation = z > 2790 ? rand()*background * 0.05 : 0.0

    return background + perturbation
end

