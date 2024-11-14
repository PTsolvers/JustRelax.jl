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

function init_rheologies(; linear=false)
   
    η_reg   = 1e18
    C       = linear ? Inf : 15e6
    ϕ       = 30
    soft_C  = NonLinearSoftening(; ξ₀=C, Δ = C / 2)       # nonlinear softening law
    # soft_C  = NonLinearSoftening()       # nonlinear softening law
    pl      = DruckerPrager_regularised(; C=C, ϕ=ϕ, η_vp=η_reg, Ψ=0.0, softening_C=soft_C)
    β       = 1e-11
    el      = ConstantElasticity(; G = 60e9, Kb = 1 / β)
    Cp      = 1050.0
    
    disl_creep  = DislocationCreep(;
        A = 1.67e-24,
        n = 3.5,
        V = 0,
        E = 1.87e5,
        R = 8.314,
    )
    g = 0
    # Define rheolgy struct
    rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.650e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_creep, el, pl)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1e-6),
            Gravity           = ConstantGravity(; g=g),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.650e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_creep, el, pl)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1e-6),
            Gravity           = ConstantGravity(; g=g),
        ),

        # Name              = "magma chamber",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=2.650e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_creep, el, pl)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1e-6),
            Gravity           = ConstantGravity(; g=g),
        ),
        # Name              = "magma chamber - hot anomaly",
        SetMaterialParams(;
            Phase             = 4,
            Density           = PT_Density(; ρ0=2.650e3, T0=273.15, β=β),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (disl_creep, el, pl)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1e-6),
            Gravity           = ConstantGravity(; g=g),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 5,
            Density           = ConstantDensity(; ρ=0e0),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k  = 2.5),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e22), el, pl)),
            Gravity           = ConstantGravity(; g=g),
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
