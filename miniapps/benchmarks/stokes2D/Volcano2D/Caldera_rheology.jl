using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheology_nonNewtonian()
    #dislocation laws
    disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    el              = ConstantElasticity(; G = 40e9)

    lithosphere_rheology = CompositeRheology( (el, disl_wet_olivine, diff_wet_olivine))
    init_rheologies(lithosphere_rheology)
end

function init_rheology_nonNewtonian_plastic()
    #dislocation laws
    disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    # plasticity
    ϕ_wet_olivine   = asind(0.1)
    C_wet_olivine   = 1e6
    η_reg           = 1e16
    el              = ConstantElasticity(; G = 40e9, ν = 0.45)
    lithosphere_rheology = CompositeRheology(
                (
                    el,
                    disl_wet_olivine,
                    diff_wet_olivine,
                    DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
                )
            )
    init_rheologies(lithosphere_rheology)
end

function init_rheology_linear()
    el    = ConstantElasticity(; G = 40e9, ν = 0.49)
    # lithosphere_rheology = CompositeRheology( (LinearViscous(; η=1e23), ))
    lithosphere_rheology = CompositeRheology( (LinearViscous(; η=1e23), el))
    init_rheologies(lithosphere_rheology)
end

function init_rheologies(; linear=false)

    η_reg   = 1e18
    C       = linear ? Inf : 10e6
    ϕ       = 15
    soft_C  = NonLinearSoftening(; ξ₀=C, Δ = C / 2)       # nonlinear softening law
    # soft_C  = NonLinearSoftening()       # nonlinear softening law
    pl      = DruckerPrager_regularised(; C=C, ϕ=ϕ, η_vp=η_reg, Ψ=0.0, softening_C=soft_C)
    el      = ConstantElasticity(; G = 25e9, ν = 0.45)
    β       = 1 / el.Kb.val
    Cp      = 1200.0

    magma_visc = ViscosityPartialMelt_Costa_etal_2009(η=LinearMeltViscosity(A = -8.1590, B = 2.4050e+04K, T0 = -430.9606K,η0=1e3Pa*s))
    #dislocation laws
    disl_top  = SetDislocationCreep(Dislocation.dry_olivine_Karato_2003)
    # diffusion laws
    disl_bot  = SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)

    # Define rheolgy struct
    rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.7e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_top, el, pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
            Melting             = MeltingParam_Smooth3rdOrder(a=517.9,  b=-1619.0, c=1699.0, d = -597.4), #mafic melting curve
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.7e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_bot, el,  pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e22), el, pl)),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
            Gravity           = ConstantGravity(; g=9.81),
        ),

        # Name              = "magma chamber",
        SetMaterialParams(;
            Phase             = 3,
            Density           = T_Density(; ρ0=2.5e3, T0=273.15),
            Conductivity      = ConstantConductivity(; k  = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            LatentHeat        = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e18), el)),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
        ),
        # Name              = "magma chamber - hot anomaly",
        SetMaterialParams(;
            Phase             = 4,
            Density           = T_Density(; ρ0=2.5e3, T0=273.15),
            Conductivity      = ConstantConductivity(; k  = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            LatentHeat        = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e18), el, )),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
        ),

        # Name              = "Conduit",
        SetMaterialParams(;
            Phase             = 5,
            Density           = T_Density(; ρ0=1.5e3, T0=273.15),
            Conductivity      = ConstantConductivity(; k  = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity(), Q_L=350e3J/kg),
            LatentHeat        = ConstantLatentHeat(Q_L=350e3J/kg),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e18), el, )),
            Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 6,
            Density           = ConstantDensity(; ρ=1e0),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e22), el, pl)),
            Gravity           = ConstantGravity(; g=9.81),
            # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
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
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
