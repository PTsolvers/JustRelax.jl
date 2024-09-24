
function init_rheologies()
    linear_viscosity_rhy            = LinearMeltViscosity(A = -8.1590, B = 2.4050e+04K, T0 = -430.9606K)#,η0=1e0Pa*s)
    linear_viscosity_bas            = LinearMeltViscosity(A = -9.6012, B = 1.3374e+04K, T0 = 307.8043K)#, η0=1e3Pa*s)
    # Define rheolgy struct
    rheology = (
        # Name              = "Rhyolite",
        SetMaterialParams(;
            Phase             = 1,
            Density           = MeltDependent_Density(ρsolid=ConstantDensity(ρ=2700kg / m^3),ρmelt=T_Density(ρ0=2300kg / m^3)),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((linear_viscosity_rhy,)),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Basaltic_Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = MeltDependent_Density(ρsolid=ConstantDensity(ρ=2900kg/m^3),ρmelt=T_Density(ρ0=2800kg / m^3)),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=400e3J/kg),
            Conductivity      = ConstantConductivity(),
            CompositeRheology = CompositeRheology((linear_viscosity_bas,)),
            Melting           = MeltingParam_Smooth3rdOrder(),
        ),
    )
end


function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j])
            if 0e0 < depth ≤ 0.1250e3
                @cell phases[ip, i, j] = 1.0
            end

            if 0.03e3 < depth ≤ 0.09e3
                @cell phases[ip, i, j] = 2.0

            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index)
end
