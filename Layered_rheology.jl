# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true)

    # Miguelitos
    # disl_upper_crust            = DislocationCreep(A=10^-28.0 , n=4.0, E=223e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_upper_crust            = DislocationCreep(A=10^-15.40, n=3.0, E=356e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lower_crust            = DislocationCreep(A=10^-15.40, n=3.0, E=356e3, V=0.0  ,  r=0.0, R=8.3145)
    # disl_lithospheric_mantle    = DislocationCreep(A=10^-15.96, n=3.5, E=530e3, V=13e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=10^-15.81, n=3.5, E=480e3, V=10e-6,  r=0.0, R=8.3145)
    # diff_lithospheric_mantle    = DiffusionCreep(  A=10^-8.16 , n=1.0, E=375e3, V=6e-6,  r=0.0, R=8.3145)
    # diff_sublithospheric_mantle = DiffusionCreep(  A=10^-8.64 , n=1.0, E=335e3, V=4e-6,  r=0.0, R=8.3145)

    # Attila
    disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=6e-6,  r=0.0, R=8.3145)
    disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    disl_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    diff_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  r=0.0, R=8.3145)


    el_upper_crust              = SetConstantElasticity(; G=25e9, ν=0.5)                             # elastic spring
    el_lower_crust              = SetConstantElasticity(; G=25e9, ν=0.5)                             # elastic spring
    el_lithospheric_mantle      = SetConstantElasticity(; G=67e9, ν=0.5)                             # elastic spring
    el_sublithospheric_mantle   = SetConstantElasticity(; G=67e9, ν=0.5)       
    β_upper_crust              = inv(get_Kb(el_upper_crust))
    β_lower_crust              = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle      = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle   = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    cohesion  = 3e6
    friction  = asind(0.2)
    # friction  = 20.0
    pl_crust        = if is_plastic 
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
    # pl        = DruckerPrager(; C = 30e6, ϕ=friction, Ψ=0.0) # non-regularized plasticity

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
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3e3, β=β_lower_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = K_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Elasticity        = el_lower_crust,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3, β=β_lithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Elasticity        = el_lithospheric_mantle,
        ),
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.3e3, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
        ),
        # Name              = "Plume",
        SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.3e3-50, β=β_sublithospheric_mantle, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle)),
            Elasticity        = el_sublithospheric_mantle,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 6,
            Density           = ConstantDensity(; ρ=1e3),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=15.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
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
        Density           = ConstantDensity(; ρ=1e3),
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


function init_phases!(phases, particles, Lx; d=650e3, r=50e3)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            depth = -(JustRelax.@cell py[ip, i, j]) #- 45e3 
            if 0e0 ≤ depth ≤ 21e3
                JustRelax.@cell phases[ip, i, j] = 1.0

            elseif 35e3 ≥ depth > 21e3
                JustRelax.@cell phases[ip, i, j] = 2.0

            elseif 90e3 ≥ depth > 35e3
                JustRelax.@cell phases[ip, i, j] = 3.0

            elseif depth > 90e3
                JustRelax.@cell phases[ip, i, j] = 3.0

            elseif 0e0 > depth 
                JustRelax.@cell phases[ip, i, j] = 6.0

            end

            # plume
            # if ((x - Lx * 0.5)^2 + (depth - d)^2) ≤ r^2
            #     JustRelax.@cell phases[ip, i, j] = 5.0
            # end

            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && ((depth - d)^2 ≤ r^2)
                JustRelax.@cell phases[ip, i, j] = 5.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx)
end

function init_phases!(phases, particles, Lx, Ly; d=650e3, r=50e3)

    @parallel_indices (i, j, k) function init_phases!(phases, px, py, pz, index, r, Lx, Ly)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j, k]) == 0 && continue

            x     = JustRelax.@cell px[ip, i, j, k]
            y     = JustRelax.@cell py[ip, i, j, k]
            depth = abs(JustRelax.@cell pz[ip, i, j, k]) #- 45e3 
            if 0e0 ≤ depth ≤ 20e3
                JustRelax.@cell phases[ip, i, j, k] = 1.0

            elseif 35e3 ≥ depth > 20e3
                JustRelax.@cell phases[ip, i, j, k] = 2.0

            elseif 100e3 ≥ depth > 40e3
                JustRelax.@cell phases[ip, i, j, k] = 3.0

            elseif depth > 100e3
                JustRelax.@cell phases[ip, i, j, k] = 4.0

            elseif 0e0 > depth 
                JustRelax.@cell phases[ip, i, j, k] = 6.0

            end

            # # plume - circular
            # if ((x - Lx * 0.5)^2 + (y - Ly * 0.5)^2 + (depth - d)^2) ≤ r^2
            #     JustRelax.@cell phases[ip, i, j, k] = 5.0
            # end
            # plume - rectangular
            if ((x - Lx * 0.5)^2 ≤ r^2) && ((depth - d)^2 ≤ r^2)
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

firstlast(x::AbstractArray) = first(x), last(x)
firstlast(x::CuArray) = extrema(x)

function inner_limits(grid::NTuple{N, T}, dxi)  where {N,T}
    # ntuple(Val(N)) do i
    #     x1 = firstlast.(grid[i])
    #     ntuple(j -> x1[j] .+ (dxi[j] * 0.5, -dxi[j] * 0.5) , Val(N))
    # end
    ntuple(Val(N)) do i
        ntuple(j -> firstlast.(grid[i])[j], Val(N))
    end
end