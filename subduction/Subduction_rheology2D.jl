# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022



function init_rheologies()
    disl_dry_olivine   = DislocationCreep(A=2.5e-17 , n=3.5, E=532e3, V=0e0  ,  r=0.0, R=8.3145)
    disl_wet_olivine   = DislocationCreep(A=9e-20   , n=3.5, E=480e3, V=11e-6,  r=0.0, R=8.3145)
    disl_wet_quartzite = DislocationCreep(A=1.97e-17, n=2.3, E=164e3, V=0e0  ,  r=0.0, R=8.3145)
    disl_plagioclase   = DislocationCreep(A=4.8e-22 , n=3.2, E=238e3, V=0e0  ,  r=0.0, R=8.3145)
    disl_gabbro        = DislocationCreep(A=4.8e-22 , n=3.2, E=238e3, V=0e0  ,  r=0.0, R=8.3145)

    diff_dry_olivine   = DiffusionCreep(A=2.5e-17 , n=3.5, E=532e3, V=0e0  ,  r=0.0, R=8.3145)
    diff_wet_olivine   = DiffusionCreep(A=9e-20   , n=3.5, E=480e3, V=11e-6,  r=0.0, R=8.3145)
    diff_wet_quartzite = DiffusionCreep(A=1.97e-17, n=2.3, E=164e3, V=0e0  ,  r=0.0, R=8.3145)
    diff_plagioclase   = DiffusionCreep(A=4.8e-22 , n=3.2, E=238e3, V=0e0  ,  r=0.0, R=8.3145)
    diff_gabbro        = DiffusionCreep(A=4.8e-22 , n=3.2, E=238e3, V=0e0  ,  r=0.0, R=8.3145)

    ϕ_dry_olivine   = asind(0.6)
    ϕ_wet_olivine   = asind(0.1)
    ϕ_wet_quartzite = asind(0.3)
    ϕ_plagioclase   = asind(0.3)

    # common physical properties
    α     = 3e-5 # 1 / K
    Cp    = 1000 # J / kg K
    C     = 3e6  # Pa
    η_reg = 1e18

    # Define rheolgy struct
    rheology = (
        # Name              = "dry ol - lithospheric mantle",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=3.3e3, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            # CompositeRheology = CompositeRheology( (LinearViscous(η = 1e20), ) ),
            CompositeRheology = CompositeRheology( 
                    (
                        disl_dry_olivine, 
                        diff_dry_olivine,
                        DruckerPrager_regularised(; C = C, ϕ=ϕ_dry_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                    ) 
                ),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.022),
            # Elasticity        = el_upper_crust,
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # # Name              = "gabbro - oceanic lithosphere",
        # SetMaterialParams(;
        #     Phase             = 2,
        #     Density           = PT_Density(; ρ0=3e3, α = α, T0 = 273),
        #     CompositeRheology = CompositeRheology( 
        #             (
        #                 disl_gabro, 
        #                 diff_gabro,
        #                 DruckerPrager_regularised(; C = C, ϕ=ϕ_gabbro, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
        #             ) 
        #         ),
        #     RadioactiveHeat   = ConstantRadioactiveHeat(0.022),
        #     # Elasticity        = el_upper_crust,
        #     Gravity           = ConstantGravity(; g=9.81),
        # ),
        # # Name              = "lower slab",
        # SetMaterialParams(;
        #     Phase             = 3,
        #     Density           = ConstantDensity(; ρ=3.28e3),
        #     CompositeRheology = CompositeRheology( (LinearViscous(η = 1e23), ) ),
        #     # Elasticity        = el_upper_crust,
        # ),
        # Name              = "wet qtz - upper continental crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.75e3, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            # CompositeRheology = CompositeRheology( (LinearViscous(η = 1e20), ) ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_wet_quartzite, 
                    diff_wet_quartzite,
                    DruckerPrager_regularised(; C = C, ϕ=ϕ_wet_quartzite, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            RadioactiveHeat   = ConstantRadioactiveHeat(2),
            # Elasticity        = el_upper_crust,
        ),
        # Name              = "plagioclase - lower continental crust",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3e3, α = α, β = 0e0, T0 = 273),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            # CompositeRheology = CompositeRheology( (LinearViscous(η = 1e20), ) ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_plagioclase, 
                    diff_plagioclase,
                    DruckerPrager_regularised(; C = C, ϕ=ϕ_plagioclase, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.2),
            # Elasticity        = el_upper_crust,
        ),
        # # Name              = "lithosphere",
        # SetMaterialParams(;
        #     Phase             = 6,
        #     Density           = PT_Density(; ρ0=3.3e3, α = α, T0 = 273),
        #     CompositeRheology = CompositeRheology( (LinearViscous(η = 1e23), ) ),
        #     # Elasticity        = el_upper_crust,
        # ),
        # Name              = "wet ol - weak zone",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=3.3e3, α = α, β = 0e0, T0 = 273),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.022),
            HeatCapacity      = ConstantHeatCapacity(; Cp=Cp),
            Conductivity      = ConstantConductivity(; k =3 ),
            # CompositeRheology = CompositeRheology( (LinearViscous(η = 1e21), ) ),
            CompositeRheology = CompositeRheology( 
                (
                    disl_wet_olivine, 
                    diff_wet_olivine,
                    DruckerPrager_regularised(; C = 5e6, ϕ=ϕ_wet_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                ) 
            ),
            # Elasticity        = el_upper_crust,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 5,
            Density           = ConstantDensity(; ρ=1e3), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp=3e3),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=1.0),
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
    end

    return nothing
end
