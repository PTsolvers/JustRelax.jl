function init_rheologies()

    # Define rheolgy struct
    el    = ConstantElasticity(; G = 40e9, ν = 0.25)
    α  = 2.4e-5 # 1 / K
    Cp = 750    # J / kg K
    return rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 3.3e3, α = α, β = 0.0e0, T0 = 273 + 1474),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Left lithosphere",
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.365e3, α = α, β = 0.0e0, T0 = 273 + 1474),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
                # Name              = "Right lithosphere",
        SetMaterialParams(;
            Phase = 3,
            Density = PT_Density(; ρ0 = 3.365e3, α = α, β = 0.0e0, T0 = 273 + 1474),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
                # Name              = "Left crust",
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantDensity(; ρ = 3.365e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
                # Name              = "Right Crust",
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 2.9e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
                        # Name              = "Weak Zone Upper Crust",
        SetMaterialParams(;
            Phase = 6,
            Density = ConstantDensity(; ρ = 3.365e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e20),)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 7,
            Density = ConstantDensity(; ρ = 1.0e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),el)),
            HeatCapacity = ConstantHeatCapacity(; Cp = 3.0e3),
            Conductivity = ConstantConductivity(; k = 1.0),
            Gravity = ConstantGravity(; g = 9.81),
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
