# from "A benchmark comparison for mantle convection codes"; Blankenbach et al., 1989

function init_rheologies()

    # Define rheolgy struct
    rheology = (
        # Name              = "UpperCrust",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=1, α = 1, β = 0.0),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.0),
            Conductivity      = ConstantConductivity(;k=1.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1),)),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Gravity           = ConstantGravity(; g = 1e4),
        ),        
    )
end

function init_phases!(phases, particles, Lx, d, r)
    ni = size(phases)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r, Lx)
        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            @cell phases[ip, i, j] = 1.0

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index, r, Lx)
end
