function init_rheologies()
    # Define rheolgy struct
    return rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 3.2e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "Oceanic lithosphere",
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 3.3e3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 0.0e0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19),)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
end

function init_phases!(phases, particles)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, particles.coords, particles.index)
end

@parallel_indices (I...) function _init_phases!(phases, pcoords::NTuple{N, T}, index) where {N, T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        px, py = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        @index phases[ip, I...] = 1.0

        # air
        if py > 0
            @index phases[ip, I...] = 3.0
        end
        # slab
        if (px ≥ 1000.0e3 && 0 ≥ py ≥ -100.0e3) || (1100.0e3 ≥ px ≥ 1000.0e3 && 0 ≥ py ≥ -200.0e3)
            @index phases[ip, I...] = 2.0
        end
    end

    return nothing
end
