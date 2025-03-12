# from Duretz et al. 2014 - http://dx.doi.org/10.1002/2014GL060438

function init_rheologies(; is_TP_Conductivity = true)

    # Dislocation creep
    Matrix = DislocationCreep(A = 3.2e-20, n = 3.0, E = 276.0e3, V = 0.0e0, r = 0.0, R = 8.3145)
    Inclusion = DislocationCreep(A = 3.16e-26, n = 3.3, E = 186.0e3, V = 0.0e0, r = 0.0, R = 8.3145)

    K_Matrix = if is_TP_Conductivity
        # crust
        TP_Conductivity(;
            a = 1.72,
            b = 807.0e0,
            c = 350,
            d = 0.0,
        )
    else
        ConstantConductivity(k = 2.5)
    end

    K_Inclusion = if is_TP_Conductivity
        TP_Conductivity(;
            a = 1.72,
            b = 807.0e0,
            c = 350,
            d = 0.0,
        )
    else
        ConstantConductivity(k = 2.5)
    end

    # Define rheolgy struct
    return rheology = (
        # Name              = "Matrix",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(ρ = 2700),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050.0),
            Conductivity = K_Matrix,
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((Matrix,)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(ρ = 2700),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1050.0),
            Conductivity = K_Inclusion,
            ShearHeat = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((Inclusion,)),
        ),
    )
end

function init_phases!(phases, particles, Lx, Ly, d, r)
    ni = size(phases)

    @parallel_indices (I...) function init_phases!(phases, px, py, pz, index, r, Lx, Ly, d)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            depth = -(@index pz[ip, I...])
            @index phases[ip, I...] = 1.0 # matrix

            # thermal anomaly - circular
            if ((x - Lx)^2 + (y - Ly)^2 + (depth - d)^2 ≤ r^2)
                @index phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, Ly, d)
end
