using GeoParams.Diffusion
using GeoParams.Dislocation

function init_rheologies(CharDim)
    disl_dry_olivine   = SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003; V = 14.5e-6m^3 / mol)
    disl_oceanic_crust = SetDislocationCreep(Dislocation.plagioclase_An75_Ji_1993)
    # disl_oceanic_litho = SetDislocationCreep(Dislocation.plagioclase_An75_Ji_1993)
    disl_cont_crust    = SetDislocationCreep(Dislocation.wet_quartzite_Kirby_1983)

    Transform_DislocationCreep(Dislocation.wet_quartzite_Kirby_1983)

    diff_dry_olivine   = SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003; V = 14.5e-6m^3 / mol)

    ϕ_dry_olivine   = sind(20)
    C_dry_olivine   = 30e6Pa

    ϕ_oceanic_crust = sind(0)
    C_oceanic_crust = 5e6Pa

    ϕ_oceanic_litho = sind(0)
    C_oceanic_litho = 5e6Pa

    ϕ_cont_crust    = sind(20)
    C_cont_crust    = 30e6Pa
    
    soft_C  = LinearSoftening((C_oceanic_litho.val*0.95, C_oceanic_litho.val), (0.1, 0.5))
    
    # common physical properties
    α     = 3e-5 / K
    Cp    = 1000 * J / kg * K
    η_reg = 1e18Pa * s 


    # Define rheolgy struct
    rheology = (
        # Name              = "dry olivine - Hirth_Kohlstedt_2003",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, α = α, β = 0e0 / Pa, T0 = 273K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3Watt/m/K),
            CompositeRheology = CompositeRheology( 
                    (
                        LinearViscous(;η=1e19Pa*s),
                        # disl_dry_olivine, 
                        # diff_dry_olivine,
                        # ConstantElasticity(; G=5e10Pa, ν=0.5),
                        # DruckerPrager_regularised(; C = C_dry_olivine, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(6.6667e-12Watt/kg),
            # Elasticity        = ConstantElasticity(; G=5e10Pa, ν=0.5),
            Gravity           = ConstantGravity(; g=9.81m/s^2),
            CharDim           = CharDim
        ),
        # Name              = "oceanic crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, α = α, β = 0e0 / Pa, T0 = 273K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3Watt/m/K),
            CompositeRheology = CompositeRheology( 
                (
                    LinearViscous(;η=1e20Pa*s),
                    # disl_oceanic_crust,
                    # ConstantElasticity(; G=5e10Pa, ν=0.5),
                    # DruckerPrager_regularised(; C = C_oceanic_crust, ϕ = ϕ_oceanic_crust, η_vp=η_reg, Ψ=0.0, softening_C = soft_C) # non-regularized plasticity
                ) 
            ),
            RadioactiveHeat   = ConstantRadioactiveHeat(2.333e-10Watt/kg),
            # Elasticity        = ConstantElasticity(; G=5e10Pa, ν=0.5),
            CharDim           = CharDim
        ),
        # Name              = "oceanic lithosphere",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, α = α, β = 0e0 / Pa, T0 = 273K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3Watt/m/K),
            CompositeRheology = CompositeRheology( 
                    (
                        LinearViscous(;η=1e19Pa*s),
                        # disl_dry_olivine, 
                        # diff_dry_olivine,
                        # ConstantElasticity(; G=5e10Pa, ν=0.5),
                        # DruckerPrager_regularised(; C = C_dry_olivine, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(6.6667e-12Watt/kg),
            # Elasticity        = ConstantElasticity(; G=5e10Pa, ν=0.5),
            CharDim           = CharDim
        ),
        # Name              = "continental crust",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=2.7e3kg / m^3, α = α, β = 0e0 / Pa, T0 = 273K),
            RadioactiveHeat   = ConstantRadioactiveHeat(5.3571e-10Watt/kg),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3Watt/m/K),
            CompositeRheology = CompositeRheology( 
                (
                    LinearViscous(;η=1e21Pa*s),
                    # disl_cont_crust, 
                    # ConstantElasticity(; G=5e10Pa, ν=0.5),
                    # DruckerPrager_regularised(; C = C_cont_crust, ϕ = ϕ_cont_crust, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            # Elasticity        = ConstantElasticity(; G=5e10Pa, ν=0.5),
            CharDim           = CharDim
        ),
        # Name              = "continental lithosphere",
        SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.3e3kg / m^3, α = α, β = 0e0 / Pa, T0 = 273K),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3Watt/m/K),
            CompositeRheology = CompositeRheology( 
                    (
                        LinearViscous(;η=1e19Pa*s),
                        # disl_dry_olivine, 
                        # diff_dry_olivine,
                        # ConstantElasticity(; G=5e10Pa, ν=0.5),
                        # DruckerPrager_regularised(; C = C_dry_olivine, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(6.6667e-12Watt/kg),
            # Elasticity        = ConstantElasticity(; G=5e10Pa, ν=0.5),
            CharDim           = CharDim
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
            ii = I[1] + offi
            jj = I[2] + offj

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
        @cell phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
