# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true, is_TP_Conductivity=true)

    # Dislocation and Diffusion creep
    Matrix            = DislocationCreep(A=3.20e-20, n=3.0, E=154e3, V=276e3,  r=0.0, R=8.3145)
    Inclusion         = DislocationCreep(A=3.16e-26, n=3.3, E=186e3, V=6e-6,  r=0.0, R=8.3145)

    # Elasticity
    el_Matrix              = SetConstantElasticity(; G=25e9, ν=0.5)
    el_Inclusion             = SetConstantElasticity(; G=25e9, ν=0.5)
    β_Matrix               = inv(get_Kb(el_Matrix))
    β_Inclusion               = inv(get_Kb(el_Inclusion))

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

    K_Matrix = if is_TP_Conductivity
    # crust
    TP_Conductivity(;
        a = 1.72,
        b = 807e0,
        c = 350,
        d = 0.0,
    )
    else
    K_Matrix = ConstantConductivity(k = 2.5)
    end

    K_Inclusion = if is_TP_Conductivity
        TP_Conductivity(;
        a = 1.72,
        b = 807e0,
        c = 350,
        d = 0.0,
    )
    else
        ConstantConductivity(k = 2.5)
    end

    # Define rheolgy struct
    rheology = (
        # Name              = "Matrix",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(ρ = 2700),
            # Density           = PT_Density(; ρ0=2.75e3, β=β_Matrix, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1050.0),
            Conductivity      = K_Matrix,
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((Matrix, el_Matrix, pl_crust)),
            Elasticity        = el_Matrix,
            # RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(ρ = 2700),
            # Density           = PT_Density(; ρ0=3e3, β=β_Inclusion, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1050.0),
            Conductivity      = K_Inclusion,
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            CompositeRheology = CompositeRheology((Inclusion, el_Inclusion, pl_crust)),
            Elasticity        = el_Inclusion,
        ),
    )
end

function init_phases!(phases, particles, Lx, d, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx, d)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j])
            @cell phases[ip, i, j] = 1.0 # matrix

            # thermal anomaly - circular
            if ((x - Lx)^2 + (depth - d)^2 ≤ r^2)
                JustRelax.@cell phases[ip, i, j] = 2.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx, d)
end
