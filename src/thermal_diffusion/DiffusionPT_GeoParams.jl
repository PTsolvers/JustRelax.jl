## Phases

@inline get_phase(x::PhaseRatios) = x.center
@inline get_phase(x) = x

# update_pt_thermal_arrays!(::Vararg{Any,N}) where {N} = nothing

function update_pt_thermal_arrays!(
    pt_thermal, phase_ratios::PhaseRatios, rheology, args, _dt
)
    ni = size(phase_ratios.center)

    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        phase_ratios.center,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        _dt,
    )

    return nothing
end

@inline function compute_phase(fn::F, rheology, phase::Int, args) where {F}
    return fn(rheology, phase, args)
end

@inline function compute_phase(fn::F, rheology, phase::Int) where {F}
    return fn(rheology, phase, args)
end

@inline function compute_phase(fn::F, rheology, phase::SVector, args) where {F}
    return fn_ratio(fn, rheology, phase, args)
end

@inline function compute_phase(fn::F, rheology, phase::SVector) where {F}
    return fn_ratio(fn, rheology, phase)
end

@inline compute_phase(fn::F, rheology, ::Nothing, args) where {F} = fn(rheology, args)
@inline compute_phase(fn::F, rheology, ::Nothing) where {F} = fn(rheology)

@inline Base.@propagate_inbounds function getindex_phase(
    phase::AbstractArray, I::Vararg{Int,N}
) where {N}
    return phase[I...]
end

@inline getindex_phase(::Nothing, I::Vararg{Int,N}) where {N} = nothing

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

# ρ*Cp

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
           compute_phase(compute_density, rheology, phase, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing,Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end

@inline function compute_ρCp(rheology, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) *
           fn_ratio(compute_density, rheology, phase_ratios, args)
end

@inline function compute_ρCp(rheology, ρ, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) * ρ
end

# α

function compute_α(rheology, phase::SArray)
    return fn_ratio(get_α, rheology, phase)
end

function compute_α(rheology, phase::Union{Int,Nothing})
    return compute_phase(get_α, rheology, phase)
end
