# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(CharDim; is_plastic = true)

    # Dislocation and Diffusion creep
    disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=6e-6,  r=0.0, R=8.3145)
    disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    diff_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  r=0.0, R=8.3145)

    # Elasticity
    el_upper_crust              = SetConstantElasticity(; G=25e9, ν=0.5)
    el_lower_crust              = SetConstantElasticity(; G=25e9, ν=0.5)
    el_lithospheric_mantle      = SetConstantElasticity(; G=67e9, ν=0.5)
    el_sublithospheric_mantle   = SetConstantElasticity(; G=67e9, ν=0.5)
    β_upper_crust               = inv(get_Kb(el_upper_crust))
    β_lower_crust               = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle       = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle    = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    cohesion  = 3e6
    friction  = asind(0.2)
    pl_crust  = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end
    friction  = asind(0.3)
    pl        = if is_plastic
        DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end
    pl_wz     = if is_plastic
        DruckerPrager_regularised(; C = 2e6, ϕ=2.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    else
        DruckerPrager_regularised(; C = Inf, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    end

    # crust
    K_crust = TP_Conductivity(;
        a = 0.64,
        b = 807e0,
        c = 0.77,
        d = 0.00004*1e-6,
    )

    K_mantle = TP_Conductivity(;
        a = 0.73,
        b = 1293e0,
        c = 0.77,
        d = 0.00004*1e-6,
    )

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.75e3, β=β_upper_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
            CharDim           = CharDim,
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3e3, β=β_lower_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            Conductivity      = K_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Elasticity        = el_lower_crust,
            CharDim           = CharDim,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3, β=β_lithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Elasticity        = el_lithospheric_mantle,
            CharDim           = CharDim,
        ),
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.3e3-50, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
            CharDim           = CharDim,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 5,
            Density           = ConstantDensity(; ρ=1e3), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=3e3),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e19),)),
            CharDim           = CharDim,
        ),
    )
end


function init_phases!(phases, particles, Lx, d, r, thick_air, CharDim)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx, CharDim)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j]) - nondimensionalize(thick_air * km, CharDim)
            if nondimensionalize(0e0km, CharDim) ≤ depth ≤ nondimensionalize(21e3km, CharDim)
                @cell phases[ip, i, j] = 1.0

            elseif nondimensionalize(35e3km, CharDim) ≥ depth >  nondimensionalize(21e3km, CharDim)
                @cell phases[ip, i, j] = 2.0

            elseif nondimensionalize(90e3km, CharDim) ≥ depth >  nondimensionalize(35e3km, CharDim)
                @cell phases[ip, i, j] = 3.0

            elseif depth > nondimensionalize(90e3km, CharDim)
                @cell phases[ip, i, j] = 3.0

            elseif depth < nondimensionalize(0e0km, CharDim)
                @cell phases[ip, i, j] = 5.0

            end 

            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && (((JustRelax.@cell py[ip, i, j]) - d - nondimensionalize(thick_air * km, CharDim))^2 ≤ r^2)
                JustRelax.@cell phases[ip, i, j] = 4.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, CharDim)
end
   