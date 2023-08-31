function init_rheologies(;)

    disl_upper_crust = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    # disl_lower_crust = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=6e-6,  r=0.0, R=8.3145)
    el_upper_crust   = SetConstantElasticity(; G=25e9, ν=0.45)
    el_lower_crust   = SetConstantElasticity(; G=25e9, ν=0.45)
    β_upper_crust    = inv(get_Kb(el_upper_crust))
    # β_lower_crust    = inv(get_Kb(el_lower_crust))
  
    # Physical properties using GeoParams ----------------
    η_reg     = 1e18
    cohesion  = 30e6
    friction  = 30
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity

    # crust
    K_crust = TP_Conductivity(;
        a = 0.64,
        b = 807e0,
        c = 0.77,
        d = 0.00004*1e-6,
    )

    visc_magma = LinearViscous(; η=1e19)
    visc_rock = LinearViscous(; η=1e24)
    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.75e3, β=β_upper_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = K_crust,
            # CompositeRheology = CompositeRheology((visc_rock, el_upper_crust, pl)),
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, )),
            # CompositeRheology = CompositeRheology((visc_rock, )),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # # Name              = "Magma",
        # SetMaterialParams(;
        #     Phase             = 2,
        #     Density           = PT_Density(; ρ0=2.75e3, β=β_lower_crust, T0=0.0, α = 3.5e-5),
        #     HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
        #     Conductivity      = K_crust,
        #     RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
        #     # CompositeRheology = CompositeRheology((visc_magma, )),
        #     CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl)),
        #     # CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, )),
        #     Elasticity        = el_lower_crust,
        # ),
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.75e3, β=β_upper_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = K_crust,
            # CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl)),
            CompositeRheology = CompositeRheology((visc_magma, )),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # # Name              = "StickyAir",
        # SetMaterialParams(;
        #     Phase             = 6,
        #     Density           = ConstantDensity(; ρ=2e3),
        #     HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
        #     RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
        #     Conductivity      = ConstantConductivity(; k=15.0),
        #     CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
        #     # Elasticity        = SetConstantElasticity(; G=Inf, ν=0.5) ,
        # ),
    )
end

function init_phases!(phases, particles, Lx; d=75e3, r=5e3)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j]) 

            # plume
            if ((x - Lx * 0.5)^2 + (depth - d)^2) ≤ r^2
                JustRelax.@cell phases[ip, i, j] = 2.0
            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end

            # # plume - rectangular
            # if ((x - Lx * 0.5)^2 ≤ r^2) && ((depth - d)^2 ≤ r^2)
            #     JustRelax.@cell phases[ip, i, j] = 5.0
            # end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx)
    
    return nothing
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