using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheologies(; ρbg = 0e0)
    #dislocation laws
    disl_dry_olivine         = SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003)
    disl_oceanic_crust_upper = SetDislocationCreep(Dislocation.wet_quartzite_Kirby_1983)
    disl_oceanic_crust_lower = SetDislocationCreep(Dislocation.plagioclase_An75_Ji_1993)
    disl_sediments           = SetDislocationCreep(Dislocation.wet_quartzite_Kirby_1983)
    disl_wet_olivine         = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)

    # diffusion laws
    diff_dry_olivine         = SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003)
    diff_wet_olivine         = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    diff_oceanic_crust_upper = DiffusionCreep(
        n=1.0NoUnits,               # power-law exponent
        r=0.0NoUnits,               # exponent of water-fugacity
        p=-3.0NoUnits,              # grain size exponent
        A=5.0716582158755556e-18,   # material specific rheological parameter
        E=154000.0,                 # activation energy
        V=0.0,   
    )
    diff_oceanic_crust_lower = DiffusionCreep( 
        n=1.0NoUnits,               # power-law exponent
        r=0.0NoUnits,               # exponent of water-fugacity
        p=-3.0NoUnits,              # grain size exponent
        A=2.063230516450235e-23,   # material specific rheological parameter
        E=238000.0,                 # activation energy
        V=0.0,   
    )
    diff_sediments           = DiffusionCreep(
        n=1.0NoUnits,               # power-law exponent
        r=0.0NoUnits,               # exponent of water-fugacity
        p=-3.0NoUnits,              # grain size exponent
        A=5.0716582158755556e-18,   # material specific rheological parameter
        E=154000.0,                 # activation energy
        V=0.0,
    )

    ϕ_dry_olivine   = asind(0.6)
    C_dry_olivine   = 3e6 

    ϕ_wet_olivine   = asind(0.1)
    C_wet_olivine   = 1e6 
    
    ϕ_oceanic_crust_upper = asind(0.1)
    C_oceanic_crust_upper = 0.3e6 
    
    ϕ_oceanic_crust_lower = asind(0.6)
    C_oceanic_crust_lower = 0.3e6 

    ϕ_sediments = asind(0.1)
    C_sediments =  0.3e6 
    
    # soft_C  = LinearSoftening((C_oceanic_litho*0.05, C_oceanic_litho), (0.1, 0.5))
    
    elasticity = ConstantElasticity(; G=5e10, ν=0.4)
    # common physical properties
    α     = 3e-5 # 1 / K
    Cp    = 1000 # J / kg K
    # C     = 3e6  # Pa
    η_reg = 1e20
    # ρbg   = 2700 # kg / m^3

    # Define rheolgy struct
    rheology = (
        # Name              = "Asthenoshpere dry olivine - Hirth_Kohlstedt_2003",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=3.3e3-ρbg, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3),
            CompositeRheology = CompositeRheology( 
                    (
                        disl_dry_olivine, 
                        diff_dry_olivine,
                        elasticity,
                        DruckerPrager_regularised(; C = C_dry_olivine, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(6.6667e-12),
            Elasticity        = elasticity,
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "sediments",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3e3-ρbg, α = α, β = 0e0, T0 = 273),
            RadioactiveHeat   = ConstantRadioactiveHeat(5.3571e-10),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_oceanic_crust_upper,
                    diff_oceanic_crust_upper,
                    elasticity,
                    DruckerPrager_regularised(; C = C_oceanic_crust_upper, ϕ = ϕ_oceanic_crust_upper, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            Elasticity        = elasticity,
        ),
        # Name              = "upper oceanic crust",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3e3-ρbg, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp= Cp),
            Conductivity      = ConstantConductivity(; k = 3 ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_oceanic_crust_upper,
                    diff_oceanic_crust_upper,
                    elasticity,
                    DruckerPrager_regularised(; C = C_oceanic_crust_upper, ϕ = ϕ_oceanic_crust_upper, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            RadioactiveHeat   = ConstantRadioactiveHeat(2.333e-10),
        ),
        # Name              = "lower oceanic lithosphere",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3e3-ρbg, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k = 3),
            CompositeRheology = CompositeRheology( 
                    (
                        disl_dry_olivine, 
                        diff_dry_olivine,
                        elasticity,
                        DruckerPrager_regularised(; C = C_dry_olivine, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(6.6667e-12),
            Elasticity        = elasticity,
        ),
        # Name              = "weak zone",
        SetMaterialParams(;
            Phase             = 5,
            Density           = PT_Density(; ρ0=3.2e3-ρbg, α = α, β = 0e0, T0 = 273),
            RadioactiveHeat   = ConstantRadioactiveHeat(5.3571e-10),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_wet_olivine, 
                    diff_wet_olivine,
                    elasticity,
                    DruckerPrager_regularised(; C = C_wet_olivine, ϕ=ϕ_wet_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            Elasticity        = elasticity,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 5,
            Density           = ConstantDensity(; ρ=1-ρbg), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=3e3),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=3.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e21),)),
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

        if pᵢ[2] ≥ 0.0 
            @cell phases[ip, I...] = 6.0
        end

    end

    return nothing
end
