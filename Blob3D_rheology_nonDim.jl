# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(CharDim)

        η_bg   = 1e20*Pa*s
        η_blob = 1e19*Pa*s
        
        # thermal conductivity
        K_crust = TP_Conductivity(;
        a = 0.64Watt / K / m ,
        b = 807e00Watt / m ,
        c = 0.77K,
        d = 0.00004/ MPa,
    )
    
        # Define rheolgy struct
        rheology = (
            # Name              = "UpperCrust",
            SetMaterialParams(;
                Phase             = 1,
                Density           = T_Density(; ρ0=3e3kg / m^3, T0=273K, α = 3.5e-5/K),
                HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2J / kg / K),
                Conductivity      = K_crust,
                CompositeRheology = CompositeRheology((LinearViscous(; η=η_bg), )),
                # Elasticity        = SetConstantElasticity(; G=25e9, ν=0.5),
                Gravity           = ConstantGravity(; g=9.81m/s^2),
                CharDim           = CharDim,
            ),
            # Name              = "LowerCrust",
            SetMaterialParams(;
                Phase             = 2,
                Density           = T_Density(; ρ0=3e3kg / m^3, T0=273K, α = 3.5e-5/K),
                HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2J / kg / K),
                Conductivity      = K_crust,
                CompositeRheology = CompositeRheology((LinearViscous(; η = η_blob), )),
                CharDim           = CharDim,
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
                depth = JustRelax.@cell pz[ip, I...]
    
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