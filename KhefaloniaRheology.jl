# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies(; is_plastic = true)

    # Dislocation and Diffusion creep
    disl_upper_crust            = DislocationCreep(A=5.07e-18, n=2.3, E=154e3, V=6e-6,  r=0.0, R=8.3145)
    disl_lower_crust            = DislocationCreep(A=2.08e-23, n=3.2, E=238e3, V=6e-6,  r=0.0, R=8.3145)
    # disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    # disl_sublithospheric_mantle = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    # diff_lithospheric_mantle    = DiffusionCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  p=0, r=0.0, R=8.3145)
    # diff_sublithospheric_mantle = DiffusionCreep(A=2.51e-17, n=1.0, E=530e3, V=6e-6,  p=0, r=0.0, R=8.3145)

    # Elasticity
    el_upper_crust              = SetConstantElasticity(; G=25e9, ν=0.5)
    el_lower_crust              = SetConstantElasticity(; G=25e9, ν=0.5)
    # el_lithospheric_mantle      = SetConstantElasticity(; G=67e9, ν=0.5)
    # el_sublithospheric_mantle   = SetConstantElasticity(; G=67e9, ν=0.5)
    β_upper_crust               = inv(get_Kb(el_upper_crust))
    β_lower_crust               = inv(get_Kb(el_lower_crust))
    # β_lithospheric_mantle       = inv(get_Kb(el_lithospheric_mantle))
    # β_sublithospheric_mantle    = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    cohesion  = 15e6
    friction  = 20
    pl_crust  = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    
    # crust
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
            Density           = PT_Density(; ρ0=2.75e3, β=β_upper_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            # Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3e3, β=β_lower_crust, T0=0.0, α = 3.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            # Conductivity      = K_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Elasticity        = el_lower_crust,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=1e2), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            # Conductivity      = ConstantConductivity(; k=1e-2),
            Conductivity      = ConstantConductivity(; k=50),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e19),)),
        ),
        # Name              = "Watah",
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ=1e3), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            # Conductivity      = ConstantConductivity(; k=1e-2),
            Conductivity      = ConstantConductivity(; k=50),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e19),)),
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

    ni = size(phases)

    for ip in JustRelax.cellaxes(phases)
        # quick escape
        @cell(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i 
            @cell pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii, jj = I[1] + offi, I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii], 
                xvi[2][jj], 
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        JustRelax.@cell phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end