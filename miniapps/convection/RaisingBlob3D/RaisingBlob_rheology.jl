# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies()

    η_bg   = 1e20
    η_blob = 1e19
    
    # thermal conductivity
    K_crust = TP_Conductivity(;
        a = 0.64,
        b = 807e0,
        c = 0.77,
        d = 0.00004*1e-6,
    )

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=3e3, β=0, T0=273, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((LinearViscous(; η=η_bg), )),
            # Elasticity        = SetConstantElasticity(; G=25e9, ν=0.5),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3e3, β=0, T0=273, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((LinearViscous(; η = η_blob), )),
            # Elasticity        = SetConstantElasticity(; G=25e9, ν=0.5),
        ),
    )
end

function init_phases!(phases, particles, Lx, Ly; d=650e3, r=50e3)
    ni = size(phases)

    @parallel_indices (I...) function init_phases!(phases, px, py, pz, index, r, Lx, Ly)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue

            x = JustRelax.@cell px[ip, I...]
            y = JustRelax.@cell py[ip, I...]
            depth = -(JustRelax.@cell pz[ip, I...])

            @cell phases[ip, I...] = 1.0

            # plume - spherical
            if ((x - Lx * 0.5)^2 + (y - Ly * 0.5)^2 + (depth - d)^2) ≤ r^2
                JustRelax.@cell phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, Ly)
    
    return nothing
end
