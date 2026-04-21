## GeoParams

# Diffusivity
"""
    compute_diffusivity(rheology, args)

Return thermal diffusivity as `k / (ρ * Cp)` for the thermodynamic state in
`args`.

This is the JustRelax wrapper around the GeoParams conductivity, heat-capacity,
and density accessors.
"""
@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

"""
    compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)

Return thermal diffusivity for a single material phase, or for the default
material when `phase === nothing`.
"""
@inline function compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) * inv(
        compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args)
    )
end

"""
    compute_diffusivity(rheology, ρ, args)

Return thermal diffusivity using an externally supplied density `ρ`.
"""
@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * ρ)
end

"""
    compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)

Return thermal diffusivity using an externally supplied density `ρ` and a
single material phase.
"""
@inline function compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) *
        inv(compute_heatcapacity(rheology, phase, args) * ρ)
end

"""
    compute_diffusivity(rheology, phase_ratios::SArray, args)

Return phase-weighted thermal diffusivity for a multi-material rheology and a
vector of phase ratios.
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

Return the volumetric heat capacity `ρ * Cp` for the thermodynamic state in
`args`.
"""
@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

"""
    compute_ρCp(rheology, phase::Union{Nothing, Int}, args)

Return volumetric heat capacity for a single material phase, or for the default
material when `phase === nothing`.
"""
@inline function compute_ρCp(rheology, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
        compute_phase(compute_density, rheology, phase, args)
end

"""
    compute_ρCp(rheology, phase_ratios::SArray, args)

Return phase-weighted volumetric heat capacity for a multi-material rheology.
"""
@inline function compute_ρCp(rheology, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) *
        fn_ratio(compute_density, rheology, phase_ratios, args)
end

"""
    compute_ρCp(rheology, ρ, args)

Return volumetric heat capacity using an externally supplied density `ρ`.
"""
@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

"""
    compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)

Return volumetric heat capacity using an externally supplied density `ρ` and a
single material phase.
"""
@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end
"""
    compute_ρCp(rheology, ρ, phase_ratios::SArray, args)

Return phase-weighted volumetric heat capacity using an externally supplied
density `ρ`.
"""
@inline function compute_ρCp(rheology, ρ, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) * ρ
end
