using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheology_nonNewtonian(CharDim)
    #dislocation laws
    disl_wet_olivine = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)

    el = ConstantElasticity(; G = 40.0e9)

    lithosphere_rheology = CompositeRheology((el, disl_wet_olivine, diff_wet_olivine))
    return init_rheologies(lithosphere_rheology, CharDim)
end

function init_rheology_nonNewtonian_plastic(CharDim)
    #dislocation laws
    disl_wet_olivine = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
    # diffusion laws
    diff_wet_olivine = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
    # plasticity
    ϕ_wet_olivine = asind(0.1)
    C_wet_olivine = 1.0e6 * Pa
    η_reg         = 1.0e20 * Pa * s
    el = ConstantElasticity(; G = 40e0GPa, ν = 0.25)
    lithosphere_rheology = CompositeRheology(
        (
            el,
            disl_wet_olivine,
            diff_wet_olivine,
            DruckerPrager_regularised(; C = C_wet_olivine, ϕ = ϕ_wet_olivine, η_vp = η_reg, Ψ = 0.0), # non-regularized plasticity
        )
    )
    return init_rheologies(lithosphere_rheology, CharDim)
end

function init_rheology_linear(CharDim)
    el = ConstantElasticity(; G = 40e0GPa, ν = 0.25)

    # lithosphere_rheology = CompositeRheology( (LinearViscous(; η=1e23), ))
    lithosphere_rheology = CompositeRheology((LinearViscous(; η = 1.0e23Pa*s), el))
    return init_rheologies(lithosphere_rheology, CharDim)
end

function init_rheologies(lithosphere_rheology, CharDim)
    # common physical properties
    α  = 2.4e-5 / K
    Cp = 750    * J / kg / K
    g  = 9.81m/s^2
    # Define rheolgy struct
    return rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 3.2e3 * kg / m^3),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k = 2.5Watt / m / K),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20Pa*s),)),
            Gravity           = ConstantGravity(; g = g),
            CharDim           = CharDim,
        ),
        # Name              = "Oceanic lithosphere",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0 = 3.2e3kg/m^3, α = α, β = 0e0/GPa, T0 = (273 + 1474)K),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k = 2.5Watt / m / K),
            Gravity           = ConstantGravity(; g = g),
            CompositeRheology = lithosphere_rheology,
            CharDim           = CharDim,
        ),
        # Name              = "oceanic crust",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ = 3.2e3 * kg / m^3),
            HeatCapacity      = ConstantHeatCapacity(; Cp = Cp),
            Conductivity      = ConstantConductivity(; k = 2.5Watt / m / K),
            Gravity           = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20Pa*s),)),
            CharDim           = CharDim,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ = 100 * kg / m^3), # water density
            HeatCapacity      = ConstantHeatCapacity(; Cp = 3.0e3 * J / kg / K),
            Conductivity      = ConstantConductivity(; k = 1.0Watt / m / K),
            Gravity           = ConstantGravity(; g = g),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19Pa*s),)),
            CharDim           = CharDim,
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
