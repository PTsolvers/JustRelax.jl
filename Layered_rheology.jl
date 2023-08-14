# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true)
    # shit from Sird
    # disl_upper_crust            = DislocationCreep(A=10^-15.0 , n=2.0, E=476e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lower_crust            = DislocationCreep(A=2.06e-23 , n=3.2, E=238e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lithospheric_mantle    = DislocationCreep(A=1.1e-16  , n=3.5, E=530e3, V=17e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=1.1e-16  , n=3.5, E=530e3, V=20e-6,  r=0.0, R=8.3145)
    # diff_lithospheric_mantle    = DiffusionCreep(A=2.46e-16   , n=1.0, E=375e3, V=10e-6,  r=0.0, R=8.3145)
    # diff_sublithospheric_mantle = DiffusionCreep(A=2.46e-16   , n=1.0, E=375e3, V=10e-6,  r=0.0, R=8.3145)

    # Miguelitos
    # disl_upper_crust            = DislocationCreep(A=10^-28.0 , n=4.0, E=223e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_upper_crust            = DislocationCreep(A=10^-15.40, n=3.0, E=356e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=10^-15.40, n=3.0, E=356e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lithospheric_mantle    = DislocationCreep(A=10^-15.96, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A=10^-15.81, n=3.5, E=480e3, V=10e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DiffusionCreep(  A=10^-8.16 , n=1.0, E=375e3, V=6e-6,  r=0.0, R=8.3145)
    diff_sublithospheric_mantle = DiffusionCreep(  A=10^-8.64 , n=1.0, E=335e3, V=4e-6,  r=0.0, R=8.3145)

    # Attila
    disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=10^-15.81, n=3.5, E=480e3, V=10e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DiffusionCreep(  A=10^-8.16 , n=1.0, E=375e3, V=6e-6,  r=0.0, R=8.3145)
    diff_sublithospheric_mantle = DiffusionCreep(  A=10^-8.64 , n=1.0, E=335e3, V=4e-6,  r=0.0, R=8.3145)


    # Burov
    # disl_upper_crust            = DislocationCreep(A=10^-14.96, n=2.25, E=212e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lower_crust            = DislocationCreep(A=10^-14.96, n=2.25, E=212e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lithospheric_mantle    = DislocationCreep(A=10^-15.96, n=3.5 , E=530e3, V=13e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=10^-19.04, n=3.5 , E=480e3, V=1.1e-5, r=1.2, R=8.3145)
    # diff_lithospheric_mantle    = DiffusionCreep(  A=10^-8.16 , n=1.0 , E=375e3, V=6e-6,   r=0.0, R=8.3145)
    # diff_sublithospheric_mantle = DiffusionCreep(  A=10^-16.6 , n=1.0 , E=375e3, V=2e-5,   r=1.0, R=8.3145)

    el_upper_crust              = SetConstantElasticity(; G=36e9, ν=0.5)                             # elastic spring
    el_lower_crust              = SetConstantElasticity(; G=40e9, ν=0.5)                             # elastic spring
    el_lithospheric_mantle      = SetConstantElasticity(; G=74e9, ν=0.5)                             # elastic spring
    el_sublithospheric_mantle   = SetConstantElasticity(; G=74e9, ν=0.5)       
    β_upper_crust              = inv(get_Kb(el_upper_crust))
    β_lower_crust              = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle      = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle   = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    G0        = 30e9    # shear modulus
    cohesion  = 20e6
    # friction  = asind(0.01)
    friction  = 20.0
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
    # pl        = DruckerPrager(; C = 30e6, ϕ=friction, Ψ=0.0) # non-regularized plasticity

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.7e3, β=β_upper_crust, T0=0.0, α = 2.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl)),
            Elasticity        = el_upper_crust,
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.9e3, β=β_lower_crust, T0=0.0, α = 2.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl)),
            Elasticity        = el_lower_crust,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3, β=β_lithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Elasticity        = el_lithospheric_mantle,
        ),
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.4e3, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
        ),
        # Name              = "Plume",
        SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.4e3-50, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 6,
            Density           = ConstantDensity(; ρ=2e3),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=15.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
            # Elasticity        = SetConstantElasticity(; G=Inf, ν=0.5) ,
        ),
    )
end

function init_rheologies_isoviscous()
    disl_upper_crust            = LinearViscous(; η=1e22)
    disl_lower_crust            = LinearViscous(; η=1e21)
    disl_lithospheric_mantle    = LinearViscous(; η=1e20)
    disl_sublithospheric_mantle = LinearViscous(; η=1e20)
    diff_lithospheric_mantle    = LinearViscous(; η=1e20)
    diff_sublithospheric_mantle = LinearViscous(; η=1e20)

    # Physical properties using GeoParams ----------------
    η_reg     = 1e18
    G0        = 30e9    # shear modulus
    cohesion  = 20e6
    # friction  = asind(0.01)
    friction  = 20.0
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5)                             # elastic spring
    β         = inv(get_Kb(el))
    ρ =  PT_Density(; ρ0=3.3e3, β=β, T0=0.0, α = 3e-5)
    upper_crust = SetMaterialParams(;
        Phase             = 1,
        Density           = PT_Density(; ρ0=2.7e3, β=β, T0=0.0, α = 2.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
        Conductivity      = ConstantConductivity(; k=2.5),
        CompositeRheology = CompositeRheology((disl_upper_crust, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=9.81),
    )
    # Name              = "LowerCrust",
    lower_crust = SetMaterialParams(;
        Phase             = 2,
        Density           = PT_Density(; ρ0=2.9e3, β=β, T0=0.0, α = 2.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
        Conductivity      = ConstantConductivity(; k=2.5),
        CompositeRheology = CompositeRheology((disl_lower_crust, )),
        Elasticity        = el,
    )
    # Name              = "LithosphericMantle",
    litho_mantle = SetMaterialParams(;
        Phase             = 3,
        Density           = PT_Density(; ρ0=3.3e3, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
        Conductivity      = ConstantConductivity(; k=3.5),
        CompositeRheology = CompositeRheology((disl_lithospheric_mantle,)),
        # Elasticity        = el,
    )
    # Name              = "SubLithosphericMantle",
    sublitho_mantle = SetMaterialParams(;
        Phase             = 4,
        Density           = PT_Density(; ρ0=3.4e3, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
        Conductivity      = ConstantConductivity(; k=3.5),
        CompositeRheology = CompositeRheology((disl_sublithospheric_mantle,)),
        # Elasticity        = el,
    )
    # Name              = "Plume",
    plume = SetMaterialParams(;
        Phase             = 5,
        Density           = PT_Density(; ρ0=3.4e3-0, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
        Conductivity      = ConstantConductivity(; k=3.5),
        CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, )),
        # Elasticity        = el,
    )
    # Name              = "StickyAir",
    sticky_air = SetMaterialParams(;
        Phase             = 6,
        Density           = ConstantDensity(; ρ=2e3),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
        Conductivity      = ConstantConductivity(; k=15.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
        # Elasticity        = SetConstantElasticity(; G=Inf, ν=0.5) ,
    )

    # Define rheolgy struct
    rheology = upper_crust, lower_crust, litho_mantle, sublitho_mantle, plume, sticky_air
    # rheology = upper_crust, upper_crust, upper_crust, upper_crust, upper_crust, upper_crust

    return rheology
end

function init_rheologies_simple(; is_plastic = true)
  
    # Attila
    disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=0.0  ,  r=0.0, R=8.3145)
    disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=10^-15.81, n=3.5, E=480e3, V=10e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DiffusionCreep(  A=10^-8.16 , n=1.0, E=375e3, V=6e-6,  r=0.0, R=8.3145)
    diff_sublithospheric_mantle = DiffusionCreep(  A=10^-8.64 , n=1.0, E=335e3, V=4e-6,  r=0.0, R=8.3145)
    
    el_upper_crust              = SetConstantElasticity(; G=36e9, ν=0.5)                             # elastic spring
    el_lower_crust              = SetConstantElasticity(; G=40e9, ν=0.5)                             # elastic spring
    el_lithospheric_mantle      = SetConstantElasticity(; G=74e9, ν=0.5)                             # elastic spring
    el_sublithospheric_mantle   = SetConstantElasticity(; G=74e9, ν=0.5)       
    β_upper_crust              = inv(get_Kb(el_upper_crust))
    β_lower_crust              = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle      = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle   = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    G0        = 30e9    # shear modulus
    cohesion  = 20e6
    # friction  = asind(0.01)
    friction  = 20.0
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
    # pl        = DruckerPrager(; C = 30e6, ϕ=friction, Ψ=0.0) # non-regularized plasticity

    upper_crust = SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.7e3, β=β_upper_crust, T0=0.0, α = 2.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl)),
            Elasticity        = el_upper_crust,
            Gravity           = ConstantGravity(; g=9.81),
        )
        # Name              = "LowerCrust",
    lower_crust = SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.9e3, β=β_lower_crust, T0=0.0, α = 2.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl)),
            Elasticity        = el_lower_crust,
            Gravity           = ConstantGravity(; g=9.81),
        )
        # Name              = "LithosphericMantle",
    lithospheric_mantle = SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3, β=β_lithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, )),
            Elasticity        = el_lithospheric_mantle,
            Gravity           = ConstantGravity(; g=9.81),
        )
        # Name              = "SubLithosphericMantle",
    sublithospheric_mantle = SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.4e3, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, )),
            Elasticity        = el_sublithospheric_mantle,
            Gravity           = ConstantGravity(; g=9.81),
        )
        # Name              = "Plume",
    plume = SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.4e3-50, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
            Gravity           = ConstantGravity(; g=9.81),
        )
        # Name              = "StickyAir",
    sticky_air = SetMaterialParams(;
            Phase             = 6,
            Density           = ConstantDensity(; ρ=2e3),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=15.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
            # Elasticity        = SetConstantElasticity(; G=Inf, ν=0.5) ,
        )
    # Define rheolgy struct
    rheology = (
        lower_crust,
        lower_crust,
        lithospheric_mantle,
        sublithospheric_mantle,
        sublithospheric_mantle,
        sublithospheric_mantle,
    )

    rheology = (
        lithospheric_mantle,
        lithospheric_mantle,
        lithospheric_mantle,
        sublithospheric_mantle,
        sublithospheric_mantle,
        plume,
    )
end

function init_phases!(phases, particles::Particles, Lx; d=650e3, r=50e3)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j]) #- 45e3 
            if 0e0 ≤ depth ≤ 20e3
                JustRelax.@cell phases[ip, i, j] = 1.0

            elseif 40e3 ≥ depth > 20e3
                JustRelax.@cell phases[ip, i, j] = 2.0

            elseif 100e3 ≥ depth > 40e3
                JustRelax.@cell phases[ip, i, j] = 3.0

            elseif depth > 100e3
                JustRelax.@cell phases[ip, i, j] = 4.0

            elseif 0e0 > depth 
                JustRelax.@cell phases[ip, i, j] = 6.0

            end

            # plume
            if ((x - Lx * 0.5)^2 + (depth - d)^2) ≤ r^2
                JustRelax.@cell phases[ip, i, j] = 5.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx)
end

function init_phases!(phases, particles::Particles, Lx, Ly; d=650e3, r=50e3)

    @parallel_indices (i, j, k) function init_phases!(phases, px, py, pz, index, r, Lx, Ly)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j, k]) == 0 && continue

            x     = JustRelax.@cell px[ip, i, j, k]
            y     = JustRelax.@cell py[ip, i, j, k]
            depth = abs(JustRelax.@cell pz[ip, i, j, k]) #- 45e3 
            if 0e0 ≤ depth ≤ 20e3
                JustRelax.@cell phases[ip, i, j, k] = 1.0

            elseif 40e3 ≥ depth > 20e3
                JustRelax.@cell phases[ip, i, j, k] = 2.0

            elseif 100e3 ≥ depth > 40e3
                JustRelax.@cell phases[ip, i, j, k] = 3.0

            elseif depth > 100e3
                JustRelax.@cell phases[ip, i, j, k] = 4.0

            elseif 0e0 > depth 
                JustRelax.@cell phases[ip, i, j, k] = 6.0

            end

            # plume
            if ((x - Lx * 0.5)^2 + (y - Ly * 0.5)^2 + (depth - d)^2) ≤ r^2
                JustRelax.@cell phases[ip, i, j, k] = 5.0
            end

        end
        return nothing
    end

    ni = size(phases)
    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, Ly)
end

function dirichlet_velocities!(Vx, εbg, xvi, xci)
    lx = abs(reduce(-, extrema(xvi[1])))
    ly = abs(reduce(-, extrema(xvi[2])))
    ε_ext = εbg
    ε_conv = εbg * 120/(ly/1e3-120)
    xv = xvi[1]
    yc = xci[2]

    @parallel_indices (i,j) function dirichlet_velocities!(Vx)
        xi = xv[i] 
        yi = yc[j] 
        Vx[i, j+1] = ε_ext * (xi - lx * 0.5) * (yi > -120e3) + ε_conv * (xi - lx * 0.5) * (yi ≤ -120e3)
        return nothing
    end

    nx, ny = size(Vx)
    @parallel (1:nx, 1:ny-2) dirichlet_velocities!(Vx)
end

function dirichlet_velocities_pureshear!(Vx, Vy, εbg, xvi, xci)
    lx = abs(reduce(-, extrema(xvi[1])))
    xv, yv = xvi

    # @parallel_indices (i, j) function velocities_x!(Vx)
    #     xi = xv[i] 
    #     yi = yc[j] 
    #     Vx[i, j+1] = εbg * (xi - lx * 0.5)
    #     return nothing
    # end
    # nx, ny = size(Vx)
    # @parallel (1:nx, 1:ny-2) velocities_x!(Vx)

    Vy[:, 1]   .= εbg * abs(yv[1])
    Vx[1, :]   .= εbg * (xv[1]-lx/2)
    Vx[end, :] .= εbg * (xv[end]-lx/2)
end