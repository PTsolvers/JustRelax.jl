# from Duretz et al. 2014 - http://dx.doi.org/10.1002/2014GL060438

function init_rheologies()

    # Dislocation creep
    Matrix        = LinearViscous(; η = 1e0)
    Inclusion     = LinearViscous(; η = 1e3)

    # Define rheolgy struct
    rheology = (
        # Name              = "Matrix",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(ρ = 1),
            CompositeRheology = CompositeRheology((Matrix, )),
            Gravity           = ConstantGravity(; g = 1e0),
        ),
        # Name              = "LowerCrust",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ConstantDensity(ρ = 2),
            CompositeRheology = CompositeRheology((Inclusion, )),
            Gravity           = ConstantGravity(; g = 1e0),
        ),
    )
end

function init_phases!(phases, particles)
    ni = size(phases)

    r = 0.1

    cx = (
        0.9, 0.2, 0.5, 0.8, 0.2, 0.3, 0.6, 0.5, 0.5, 0.7,
    )
    cy = (
        0.9, 0.3, 0.3, 0.8, 0.5, 0.3, 0.4, 0.8, 0.8, 0.4,
    )
    cz = (
        0.8, 0.4, 0.7, 0.8, 0.4, 0.3, 0.8, 0.9, 0.6, 0.6,
    )

    @parallel_indices (I...) function init_phases!(phases, px, py, pz, index, r, cx, cy, cz)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            z = @index pz[ip, I...]
            @index phases[ip, I...] = 1.0 # matrix

            # thermal anomaly - circular
            for (cxᵢ, cyᵢ, czᵢ) in zip(cx, cy, cz)
                if (x - cxᵢ)^2 + (y - cyᵢ)^2 + (z - czᵢ)^2 ≤ r^2
                    @index phases[ip, I...] = 2.0
                end
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, r, cx, cy, cz)
end

