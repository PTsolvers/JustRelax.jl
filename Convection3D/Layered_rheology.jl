# from "Self-consistent generation of tectonic plates in time-dependent, three-dimensional mantle convection simulations 1. Pseudoplastic yielding" Paul J. Tackley, 2000
# https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2000GC000036
using GeoParams

@inline function custom_viscosity(
    a::CustomRheology; T = 0.0, kwargs...
)
    (; E) = a.args
    η = exp(E * (inv(T + 1) - 0.5))

    return  η
end

# function to compute deviatoric stress
@inline function custom_τII(a::CustomRheology, EpsII; kwargs...)
    η = custom_viscosity(a; kwargs...)
    return 2.0 * (η * EpsII)
end

# function to compute strain rate
@inline function custom_εII(a::CustomRheology, TauII; kwargs...)
    η = custom_viscosity(a; kwargs...)
    return (TauII / η) * 0.5
end

function init_rheologies(CharDim)

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16Pas
    cohesion  = 30MPa
    friction  = asind(0.3)
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
   
    # constant parameters, these are typically wrapped into a struct (compulsory)
    η_parameters = (; E = 23.03)
    v            = CustomRheology(custom_εII, custom_τII, η_parameters)

    # crust
    K_mantle = TP_Conductivity(;
        a = 0.73,
        b = 1293e0,
        c = 0.77,
        d = 0.00004*1e-6,
    )

    # Define rheolgy struct
    rheology = (
        # Name              = "SubLithosphericMantle",
        SetMaterialParams(;
            Phase             = 4,
            Density           = T_Density(; ρ0=3.3e3kg/m^3),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1.25e3J/kg/K),
            Conductivity      = ConstantConductivity(; k=3.96Watt/K/m),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            CompositeRheology = CompositeRheology((v, pl)),
            CharDim             = CharDim,
        ),
    )
end

function init_phases!(phases, particles)
    ni = size(phases)

    @parallel_indices (I...) function init_phases!(phases, index)

        @inbounds for ip in JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, I...]) == 0 && continue
            JustRelax.@cell phases[ip, I...] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.index)
end
