using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheologies(; linear = false, incompressible = true, isplastic = true, magma = false)

    η_reg = 1.0e15
    C = isplastic ? 10.0e6 : Inf
    ϕ = 15
    Ψ = 0.0
    soft_C = NonLinearSoftening(; ξ₀ = C, Δ = C / 1.0e5)       # nonlinear softening law
    pl = DruckerPrager_regularised(; C = C * MPa, ϕ = ϕ, η_vp = (η_reg) * Pas, Ψ = Ψ, softening_C = soft_C)
    G0 = 25.0e9Pa        # elastic shear modulus
    G_magma = 10.0e9Pa        # elastic shear modulus magma

    el = incompressible ? ConstantElasticity(; G = G0, ν = 0.45) : ConstantElasticity(; G = G0, ν = 0.25)
    el_magma = incompressible ? ConstantElasticity(; G = G_magma, ν = 0.45) : ConstantElasticity(; G = G_magma, ν = 0.25)
    β = 1 / el.Kb.val
    β_magma = 1 / el_magma.Kb.val
    Cp = 1200.0

    magma_visc = magma ? ViscosityPartialMelt_Costa_etal_2009(η = LinearMeltViscosity(A = -8.159, B = 2.405e+4K, T0 = -430.9606K)) : LinearViscous(η = 1.0e15)
    conduit_visc = magma ? ViscosityPartialMelt_Costa_etal_2009(η = LinearMeltViscosity(A = -8.159, B = 2.405e+4K, T0 = -430.9606K)) : LinearViscous(η = 1.0e15)
    #dislocation laws
    disl_top = linear ? LinearViscous(η = 1.0e23) : DislocationCreep(; A = 1.67e-24, n = 3.5, E = 1.87e5, V = 6.0e-6, r = 0.0, R = 8.3145)
    # disl_top  = SetDislocationCreep(Dislocation.dry_olivine_Karato_2003)
    # diffusion laws
    disl_bot = linear ? LinearViscous(η = 1.0e21) : SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)

    # Define rheolgy struct
    return rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((disl_top, el, pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23), el, pl)),
            Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((disl_bot, el, pl)),
            # CompositeRheology = CompositeRheology( (LinearViscous(; η=1e21), el, pl)),
            Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),

        # Name              = "magma chamber",
        SetMaterialParams(;
            Phase = 3,
            Density = MeltDependent_Density(ρsolid = PT_Density(ρ0 = 2.4e3, T0 = 273.15, β = β_magma), ρmelt = T_Density(ρ0 = 2.2e3, T0 = 273.15)),
            # Density           = PT_Density(; ρ0=2.4e3, T0=273.15, β=β_magma),
            Conductivity = ConstantConductivity(; k = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity = Latent_HeatCapacity(Cp = ConstantHeatCapacity(), Q_L = 350.0e3J / kg),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            CompositeRheology = CompositeRheology((magma_visc, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        ),
        # Name              = "magma chamber - hot anomaly",
        SetMaterialParams(;
            Phase = 4,
            # Density           = T_Density(; ρ0=2.2e3, T0=273.15),
            # Density           = BubbleFlow_Density(ρgas=ConstantDensity(ρ=10.0), ρmelt=ConstantDensity(ρ=2.4e3), c0=4e-2),
            Density = BubbleFlow_Density(ρgas = ConstantDensity(ρ = 10.0), ρmelt = MeltDependent_Density(ρsolid = T_Density(ρ0 = 2.4e3, T0 = 273.15), ρmelt = ConstantDensity(ρ = 2.2e3)), c0 = 4.0e-2),
            Conductivity = ConstantConductivity(; k = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity = Latent_HeatCapacity(Cp = ConstantHeatCapacity(), Q_L = 350.0e3J / kg),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            CompositeRheology = CompositeRheology((magma_visc, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        ),

        # Name              = "Conduit",
        SetMaterialParams(;
            Phase = 5,
            Density = BubbleFlow_Density(ρgas = ConstantDensity(ρ = 10.0), ρmelt = MeltDependent_Density(ρsolid = T_Density(ρ0 = 2.4e3, T0 = 273.15), ρmelt = ConstantDensity(ρ = 2.2e3)), c0 = 4.0e-2),
            # Density           = BubbleFlow_Density(ρgas=ConstantDensity(ρ=10.0), ρmelt=ConstantDensity(ρ=2.4e3), c0=4e-2),
            # Density           = T_Density(; ρ0=1.5e3, T0=273.15),
            Conductivity = ConstantConductivity(; k = 1.5),
            # HeatCapacity      = Latent_HeatCapacity(Cp=ConstantHeatCapacity()),
            HeatCapacity = Latent_HeatCapacity(Cp = ConstantHeatCapacity(), Q_L = 350.0e3J / kg),
            LatentHeat = ConstantLatentHeat(Q_L = 350.0e3J / kg),
            CompositeRheology = CompositeRheology((conduit_visc, el_magma)),
            Melting = MeltingParam_Smooth3rdOrder(a = 3043.0, b = -10552.0, c = 12204.9, d = -4709.0), #felsic melting curve
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 6,
            Density = ConstantDensity(; ρ = 1.0e0),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22), el, pl)),
            Gravity = ConstantGravity(; g = 9.81),
            # Melting           = MeltingParam_Smooth3rdOrder(a=3043.0,b=-10552.0,c=12204.9,d=-4709.0), #felsic melting curve
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

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
