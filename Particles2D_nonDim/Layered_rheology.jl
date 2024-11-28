using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheologies(CharDim; is_plastic = true)
    # from https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2022JB025877
    diff_ol = DiffusionCreep(
        n = 1NoUnits,
        r = 0NoUnits,
        A = 1.25e-15Pa^(-1) / s,
        E = 370.0e3J / mol,
        V = 6e-6m^3 / mol,
        R = 8.3145J / mol / K,
    )
    disl_ol = DislocationCreep(
        n = 3.5NoUnits,
        r = 0NoUnits,
        A = 8.33e-15Pa^(-35//10) / s,
        E = 530.0e3J / mol,
        V = 1.4e-5m^3 / mol,
        R = 8.3145J / mol / K,
    )

    diff_wad = DiffusionCreep(
        n = 1NoUnits,
        r = 0NoUnits,
        A = 6.12e-19Pa^(-1) / s,
        E = 231.0e3J / mol,
        V = 6e-6m^3 / mol,
        R = 8.3145J / mol / K,
    )
    disl_wad = DislocationCreep(
        n = 3.5NoUnits,
        r = 0NoUnits,
        A = 2.05e-12Pa^(-35//10) / s,
        E = 530.0e3J / mol,
        V = 1.7e-5m^3 / mol,
        R = 8.3145J / mol / K,
    )

    diff_ring = DiffusionCreep(
        n = 1NoUnits,
        r = 0NoUnits,
        A = 2.94e-17Pa^(-1) / s,
        E = 270.0e3J / mol,
        V = 6e-6m^3 / mol,
        R = 8.3145J / mol / K,
    )
    disl_ring = DislocationCreep(
        n = 3.5NoUnits,
        r = 0NoUnits,
        A = 2.05e-12Pa^(-35//10) / s,
        E = 530.0e3J / mol,
        V = 1.7e-5m^3 / mol,
        R = 8.3145J / mol / K,
    )

    diff_lower_mantle = DiffusionCreep(
        n = 1NoUnits,
        r = 0NoUnits,
        A = 5.4e-22Pa^(-1) / s,
        E = 270.0e3J / mol,
        V = 6e-6m^3 / mol,
        R = 8.3145J / mol / K,
    )
    disl_lower_mantle = DislocationCreep(
        n = 3.5NoUnits,
        r = 0NoUnits,
        A = 1e-40Pa^(-35//10) / s,
        E = 530.0e3J / mol,
        V = 0m^3 / mol,
        R = 8.3145J / mol / K,
    )

    # Dislocation and Diffusion creep
    disl_upper_crust            = SetDislocationCreep(Dislocation.dry_anorthite_Rybacki_2006)
    disl_lower_crust            = SetDislocationCreep(Dislocation.dry_anorthite_Rybacki_2006)
    disl_lithospheric_mantle    = SetDislocationCreep(Dislocation.wet_olivine_Hirth_2003)
    disl_sublithospheric_mantle = SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003)
    disl_bridgemanite           = SetDislocationCreep(Dislocation.wet_olivine_Hirth_2003)
    disl_ppv                    = SetDislocationCreep(Dislocation.wet_olivine_Hirth_2003)

    diff_upper_crust            = SetDiffusionCreep(Diffusion.dry_anorthite_Rybacki_2006)
    diff_lower_crust            = SetDiffusionCreep(Diffusion.dry_anorthite_Rybacki_2006)
    diff_lithospheric_mantle    = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    diff_sublithospheric_mantle = SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003)
    diff_bridgemanite           = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    diff_ppv                    = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    # Elasticity
    el_upper_crust              = SetConstantElasticity(; G=25e9Pa, ν=0.5)
    el_lower_crust              = SetConstantElasticity(; G=25e9Pa, ν=0.5)
    el_lithospheric_mantle      = SetConstantElasticity(; G=67e9Pa, ν=0.5)
    el_sublithospheric_mantle   = SetConstantElasticity(; G=67e9Pa, ν=0.5)
    β_upper_crust               = inv(get_Kb(el_upper_crust))
    β_lower_crust               = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle       = inv(get_Kb(el_lithospheric_mantle))
    β_sublithospheric_mantle    = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16 * Pa * s
    cohesion  = 3e6 * Pa
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

    K_crust = TP_Conductivity(;
        a = 0.64Watt / K / m ,
        b = 807e00Watt / m ,
        c = 0.77K,
        d = 0.00004/ MPa,
    )
    K_mantle = TP_Conductivity(;
        a = 0.73Watt / K / m ,
        b = 1293e00Watt / m ,
        c = 0.77K,
        d = 0.00004/ MPa,
    )

    g = 9.81m/s^2

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.75e3kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2J / kg / K),
            Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust, el_upper_crust, pl_crust)),
            Elasticity        = el_upper_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g=g),
            CharDim           = CharDim,
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.8kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2J / kg / K),
            Conductivity      = K_crust,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lower_crust, el_lower_crust, pl_crust)),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_lower_crust,
            CharDim           = CharDim,
        ),
        # Name              = "LithosphericMantle",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3J / kg / K),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el_lithospheric_mantle, pl)),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_lithospheric_mantle,
            CharDim           = CharDim,
        ),
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3J / kg / K),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el_sublithospheric_mantle, pl)),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_lithospheric_mantle,
            CharDim           = CharDim,
        ),
        # Name              = "Bridgemanite",
        SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3J / kg / K),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_bridgemanite, diff_bridgemanite, el_sublithospheric_mantle)),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_sublithospheric_mantle,
            CharDim           = CharDim,
        ),
        # Name              = "ppv",
        SetMaterialParams(;
            Phase             = 6,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, β=β_upper_crust, T0=0e0C, α = 3.5e-5/ K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3J / kg / K),
            Conductivity      = K_mantle,
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((disl_ppv, diff_ppv, el_sublithospheric_mantle)),
            Gravity           = ConstantGravity(; g=g),
            Elasticity        = el_sublithospheric_mantle,
            CharDim           = CharDim,
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{2, T}, index, xvi) where {T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(2)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:2))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{3, T}, index, xvi) where {T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(3)) do i 
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1, offk in 0:1
            ii, jj, kk = I[1] + offi, I[2] + offj, I[3] + offk

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue
            !(kk ≤ ni[3]) && continue

            xvᵢ = (
                xvi[1][ii], 
                xvi[2][jj], 
                xvi[3][kk], 
            )
            # @show xvᵢ ii jj kk
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:3))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj, kk]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end