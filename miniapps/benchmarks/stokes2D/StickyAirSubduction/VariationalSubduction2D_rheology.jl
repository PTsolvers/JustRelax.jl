function init_rheologies()
   
    # Define rheolgy struct
    rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ=3.2e3),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e21),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "Oceanic lithosphere",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(; ρ=3.3e3),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23),)),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 3,
            Density           = ConstantDensity(; ρ=3.3e3),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e23),)),
            Gravity           = ConstantGravity(; g=9.81),
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
