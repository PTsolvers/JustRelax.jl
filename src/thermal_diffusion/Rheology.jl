## GeoParams

# Diffusivity 

@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

@inline function compute_diffusivity(rheology, phase::Union{Nothing,Int}, args)
    return compute_conductivity(rheology, phase, args) * inv(
        compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args)
    )
end

@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

@inline function compute_diffusivity(rheology, ρ, phase::Union{Nothing,Int}, args)
    return compute_conductivity(rheology, phase, args) *
           inv(compute_heatcapacity(rheology, phase, args) * ρ)
end

@inline function compute_diffusivity(
    rheology::NTuple{N,AbstractMaterialParamsStruct}, phase_ratios::SArray, args
) where {N}
    ρ = compute_density_ratio(phase_ratios, rheology, args)
    conductivity = fn_ratio(compute_conductivity, rheology, phase_ratios, args)
    heatcapacity = fn_ratio(compute_heatcapacity, rheology, phase_ratios, args)
    return conductivity * inv(heatcapacity * ρ)
end

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
           compute_phase(compute_density, rheology, phase, args)
end

@inline function compute_ρCp(rheology, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) *
           fn_ratio(compute_density, rheology, phase_ratios, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) * ρ
end
