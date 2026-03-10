## GeoParams

# Diffusivity
"""
    compute_diffusivity(rheology, args)

Compute the thermal diffusivity for the thermal diffusion solver using the rheology model `rheology` and the arguments `args` (e.g., temperature, pressure, phase, etc.).
This is the JustRelax wrapper that calls the GeoParams.jl functions.
"""
@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

"""
    compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)

Compute the thermal diffusivity for the thermal diffusion solver using the rheology model `rheology`, the phase index `phase`, and the arguments `args` (e.g., temperature, pressure, etc.).
This is the JustRelax wrapper that calls the GeoParams.jl functions.
"""
@inline function compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) * inv(
        compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args)
    )
end

"""
    compute_diffusivity(rheology, ρ, args)

Computes the thermal diffusivity based on the rheology and the density with given arguments.
This is the JustRelax wrapper that calls the GeoParams.jl functions.
"""
@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * ρ)
end

"""
    compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)

Computes the thermal diffusivity based on the rheology, the density, and the phase with given arguments.
This is the JustRelax wrapper that calls the GeoParams.jl functions.
"""
@inline function compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) *
        inv(compute_heatcapacity(rheology, phase, args) * ρ)
end

"""
    compute_diffusivity(rheology, phase_ratios::SArray, args)

Compute the thermal diffusivity for the thermal diffusion solver using the rheology model `rheology`, the phase ratios `phase_ratios`, and the arguments `args`.
This is the JustRelax wrapper that calls the GeoParams.jl functions based on the phase ratios.
"""
@inline function compute_diffusivity(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, phase_ratios::SArray, args
    ) where {N}
    ρ = compute_density_ratio(phase_ratios, rheology, args)
    conductivity = fn_ratio(compute_conductivity, rheology, phase_ratios, args)
    heatcapacity = fn_ratio(compute_heatcapacity, rheology, phase_ratios, args)
    return conductivity * inv(heatcapacity * ρ)
end

# ρ*Cp
"""
    compute_ρCp(rheology, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology` and the arguments `args`.
This is the calls the GeoParams.jl functions internally.
"""
@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

"""
    compute_ρCp(rheology, phase::Union{Nothing, Int}, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology`, the phase index `phase`, and the arguments `args`.
This is the calls the GeoParams.jl functions internally based on the phase (Integer) or the phase (Nothing).
"""
@inline function compute_ρCp(rheology, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
        compute_phase(compute_density, rheology, phase, args)
end

"""
    compute_ρCp(rheology, ρ, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology`, the density `ρ`, and the arguments `args`.
This is the calls the GeoParams.jl functions internally based on the phase ratios.
"""
@inline function compute_ρCp(rheology, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) *
        fn_ratio(compute_density, rheology, phase_ratios, args)
end

"""
    compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology`, the density `ρ`, the phase index `phase`, and the arguments `args`.
This is the calls the GeoParams.jl functions internally with a given density.
"""
@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

"""
    compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology`, the density `ρ`, the phase index `phase`, and the arguments `args`.
This is the calls the GeoParams.jl functions internally with a given density and phase.
"""
@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end
"""
    compute_ρCp(rheology, ρ, phase_ratios::SArray, args)

Compute the product of density and heat capacity for the thermal diffusion solver using the rheology model `rheology`, the density `ρ`, the phase ratios `phase_ratios`, and the arguments `args`.
This is the calls the GeoParams.jl functions internally with a given density and phase ratios.
"""
@inline function compute_ρCp(rheology, ρ, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) * ρ
end
