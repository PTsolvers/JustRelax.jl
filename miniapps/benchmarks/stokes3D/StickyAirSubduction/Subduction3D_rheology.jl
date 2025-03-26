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
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj, kk]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
