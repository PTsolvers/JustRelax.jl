# from "A benchmark comparison for mantle convection codes"; Blankenbach et al., 1989

function init_rheologies()

    # Ed  =   Ea * R * ΔT                    # J / mol
    # Ar  =   1/2/η₀ * exp((Ed)/(R*Tη))      # Pa^-1 s^-1
    
    
    # Arrhenius = DiffusionCreep(
    #     A = ArPa^(-1) * s^(-1),       # material specific rheological parameter
    #     E = 375e3J / mol,                  # activation energy
    #     V = 0.0m^3 / mol,                 # activation Volume
    #     n = 1.0NoUnits,                    # power-law exponent
    #     r = 0.0NoUnits,                    # exponent of water-fugacity
    #     p = 0NoUnits,                      # grain size exponent
    #     Apparatus = Invariant,
    # )

    # Define rheolgy struct
    return rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 1, α = 1, β = 0.0),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.0),
            Conductivity = ConstantConductivity(; k = 1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1),)),
            RadioactiveHeat = ConstantRadioactiveHeat(0.0),
            Gravity = ConstantGravity(; g = 1.0e4),
        ),
    )
end

function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, index)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue
            @index phases[ip, i, j] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.index)
    return nothing
end

