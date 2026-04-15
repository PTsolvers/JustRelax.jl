# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true)

    # Dislocation and Diffusion creep
    disl_upper_crust = DislocationCreep(
        A = 5.07e-18,
        n = 2.3,
        E = 154.0e3,
        V = 8.0e-6,
        r = 0.0,
        R = 8.3145
    )
    disl_lower_crust = DislocationCreep(
        A = 2.08e-23,
        n = 3.5,
        E = 238.0e3,
        V = 8.0e-6,
        r = 0.0,
        R = 8.3145,
    )
    disl_litho = DislocationCreep( # dry olivine
        A = 1.1e-17,
        n = 3.5,
        E = 530.0e3,
        V = 13.0e-6,
        r = 0.0,
        R = 8.3145,
    )
    disl_upper_mantle = DislocationCreep( # dry olivine
        A = 1.1e-17,
        n = 3.5,
        E = 230.0e3,
        V = 11.0e-6,
        r = 0.0,
        R = 8.3145,
    )
    
    # down to 660km
    Adiff = 2.2e-10
    diff_litho = DiffusionCreep(
        n = 1.0,                         # power-law exponent
        r = 0.0,                         # exponent of water-fugacity
        p = 0,                           # grain size exponent
        A = Adiff,        # material specific rheological parameter
        E = 375e3,                      # activation energy
        V = 3.65e-6,                   # activation Volume
    )
    diff_upper_mantle = DiffusionCreep( # dry olivine
       n = 1.0,                         # power-law exponent
        r = 0.0,                         # exponent of water-fugacity
        p = 0,                           # grain size exponent
        A = Adiff,        # material specific rheological parameter
        E = 375.0e3,                      # activation energy
        V = 3.65e-6,                   # activation Volume
    )
    diff_lower_mantle = DiffusionCreep( # dry olivine
        n = 1.0,                         # power-law exponent
        r = 0.0,                         # exponent of water-fugacity
        p = 0,                           # grain size exponent
        A = Adiff,    # material specific rheological parameter
        E = 375.0e3,                      # activation energy
        V = 3.65e-6,                   # activation Volume
    )
    # Elasticity
    el_upper_crust = SetConstantElasticity(;  G = 25.0e9, ν = 0.4)
    el_lower_crust = SetConstantElasticity(;  G = 25.0e9, ν = 0.4)
    el_upper_mantle = SetConstantElasticity(; G = 60.0e9, ν = 0.4)
    el_lower_mantle = SetConstantElasticity(; G = 60.0e9, ν = 0.4)
    β_upper_crust = inv(get_Kb(el_upper_crust))
    β_lower_crust = inv(get_Kb(el_lower_crust))
    β_upper_mantle = inv(get_Kb(el_upper_mantle))
    β_lower_mantle = inv(get_Kb(el_lower_mantle))

    # Physical properties using GeoParams ----------------
    η_reg    = 1.0e18
    cohesion = 5.0e6
    friction = 5e0
    pl_crust = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    end
    cohesion = 30e6
    friction = 30e0
    pl = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ = friction, η_vp = η_reg, Ψ = 0.0) # non-regularized plasticity
    end

    K_crust = TP_Conductivity(;
        a = 0.64,
        b = 807.0e0,
        c = 0.77,
        d = 4.0e-11,
    )
    K_mantle = TP_Conductivity(;
        a = 0.73,
        b = 1293.0e0,
        c = 0.77,
        d = 4.0e-11,
    )

    g = 9.81

    # Define rheolgy struct
    return rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.82e3, β = β_upper_crust, T0 = 273e0, α = 3e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity = el_upper_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            Gravity = ConstantGravity(; g = g),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.85e3, β = β_lower_crust, T0 = 273e0, α = 3e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 7.5e2),
            Conductivity = K_crust,
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Gravity = ConstantGravity(; g = g),
            Elasticity = el_lower_crust,
        ),
        # Name              = "Down to 660",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.24e3, β = β_upper_mantle, T0 = 273e0, α = 5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((disl_litho, diff_litho, el_upper_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12),
            Elasticity = el_upper_mantle,
        ),
        # Name              = "Down to 2700"
        SetMaterialParams(;
            Phase = 4,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_upper_mantle, T0 = 273e0, α = 5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1e3),
            Conductivity = K_mantle,
            # CompositeRheology = CompositeRheology((diff_upper_mantle, el_upper_mantle, pl)),
            CompositeRheology = CompositeRheology((disl_upper_mantle, diff_upper_mantle, el_upper_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12),
            Elasticity = el_upper_mantle,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 5,
            Density = PT_Density(; ρ0 = 3.3e3, β = β_lower_mantle, T0 = 273e0, α = 5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1e3),
            Conductivity = K_mantle,
            CompositeRheology = CompositeRheology((diff_lower_mantle, el_lower_mantle, pl)),
            Gravity = ConstantGravity(; g = g),
            RadioactiveHeat = ConstantRadioactiveHeat(8.6e-12),
            Elasticity = el_lower_mantle,
        ),
    )
end

function init_phases!(phases, particles, Lx, d, r, thick_air)
    ni = size(phases)

    d_air = thick_air
    d_0km = 0.0e0
    d_21km = 21.0e3
    d_35km = 35.0e3
    d_120km = 120.0e3
    d_2700km = 2790.0e3

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

            elseif d_120km ≥ depth > d_35km
                @index phases[ip, i, j] = 3.0

            elseif d_2700km ≥ depth > d_120km
                @index phases[ip, i, j] = 4.0

            else#if depth < d_2700km
                # @index phases[ip, i, j] = 5.0
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
# Distances in this profile are expressed in kilometers.

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

    perturbation = A * sin(n * π * x / Lx) * sin(m * π * z / Lz)
    # perturbation = z > 2790 ? rand()*background * 0.05 : 0.0

    return background + perturbation
end
