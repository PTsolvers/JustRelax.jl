## Phases

@inline get_phase(x::JustPIC.PhaseRatios) = x.center
@inline get_phase(x) = x

@inline get_phase_fluxes(x::JustPIC.PhaseRatios, ::NTuple{2}) = x.Vx, x.Vy
@inline get_phase_fluxes(x, ::NTuple{2}) = x, x
@inline get_phase_fluxes(x::JustPIC.PhaseRatios, ::NTuple{3}) = x.Vx, x.Vy, x.Vz
@inline get_phase_fluxes(x, ::NTuple{3}) = x, x, x

"""
    update_pt_thermal_arrays!(pt_thermal, phase_ratios, rheology, args, _dt)

Recompute the pseudo-transient thermal coefficient arrays stored in
`pt_thermal` from phase-weighted material properties.

This helper is used by the pseudo-transient thermal solver when the local phase
mixture changes over time.
"""
function update_pt_thermal_arrays!(
        pt_thermal, phase_ratios::JustPIC.PhaseRatios, rheology, args, _dt
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
    return fn(rheology[phase])
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
        phase::AbstractArray, I::Vararg{Int, N}
    ) where {N}
    return phase[I...]
end

@inline getindex_phase(::Nothing, I::Vararg{Int, N}) where {N} = nothing

"""
    phase_at_face(phase_ratios, n_cells, dim, i_face, i_L, i_R)

Read the phase ratio(s) to combine with the two states bracketing a conductive
flux face at `i_face`, which lies between the cell-centered indices `i_L` and
`i_R` along dimension `dim`.

`phase_ratios` is either sized like the flux array along `dim` (`n_cells + 1`
entries, one per face — such as `PhaseRatios.Vx/Vy/Vz`) or like the cell-center
grid along `dim` (`n_cells` entries, shared between the two bracketing cells).
A face-sized array is read once at `i_face` and that value is reused for both
states; a cell-center-sized array is read separately at `i_L` and `i_R`. Any
other size along `dim` is a mismatch between the phase-ratio array and the grid
it is being sampled on.
"""
@inline function phase_at_face(
        phase_ratios::AbstractArray, n_cells::Int, dim::Int,
        i_face::NTuple{N, Int}, i_L::NTuple{N, Int}, i_R::NTuple{N, Int}
    ) where {N}
    n = size(phase_ratios, dim)
    if n == n_cells + 1
        phase_face = getindex_phase(phase_ratios, i_face...)
        return phase_face, phase_face
    elseif n == n_cells
        return getindex_phase(phase_ratios, i_L...), getindex_phase(phase_ratios, i_R...)
    else
        # constant message: string interpolation does not compile in GPU kernels
        throw(DimensionMismatch("phase-ratio array is neither cell-centered nor face-centered along the flux direction"))
    end
end

@inline function phase_at_face(
        ::Nothing, ::Int, ::Int, ::NTuple{N, Int}, ::NTuple{N, Int}, ::NTuple{N, Int}
    ) where {N}
    return nothing, nothing
end

# Diffusivity

@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

@inline function compute_diffusivity(rheology, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) * inv(
        compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args)
    )
end

@inline function compute_diffusivity(rheology, ρ, args)
    return compute_conductivity(rheology, args) *
        inv(compute_heatcapacity(rheology, args) * ρ)
end

@inline function compute_diffusivity(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_conductivity(rheology, phase, args) *
        inv(compute_heatcapacity(rheology, phase, args) * ρ)
end

@inline function compute_diffusivity(
        rheology::NTuple{N, AbstractMaterialParamsStruct}, phase_ratios::SArray, args
    ) where {N}
    return fn_ratio(compute_diffusivity, rheology, phase_ratios, args)
end

# ρ*Cp

@inline function compute_ρCp(rheology, args)
    return compute_heatcapacity(rheology, args) * compute_density(rheology, args)
end

@inline function compute_ρCp(rheology, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) *
        compute_phase(compute_density, rheology, phase, args)
end

@inline function compute_ρCp(rheology, ρ, args)
    return compute_heatcapacity(rheology, args) * ρ
end

@inline function compute_ρCp(rheology, ρ, phase::Union{Nothing, Int}, args)
    return compute_phase(compute_heatcapacity, rheology, phase, args) * ρ
end

@inline function compute_ρCp(rheology, phase_ratios::SArray, args)
    return fn_ratio(compute_ρCp, rheology, phase_ratios, args)
end

@inline function compute_ρCp(rheology, ρ, phase_ratios::SArray, args)
    return fn_ratio(compute_heatcapacity, rheology, phase_ratios, args) * ρ
end

# α

"""
    compute_α(rheology, phase)

Return the thermal expansivity `α` used by the adiabatic heating kernels.

`phase` can be a single phase index, `nothing`, or a phase-ratio vector. In the
latter case the result is phase-weighted.
"""
function compute_α(rheology, phase::SArray)
    return fn_ratio(get_α, rheology, phase)
end

function compute_α(rheology, phase::Union{Int, Nothing})
    return compute_phase(get_α, rheology, phase)
end

function compute_radioactive_heating(rheology, phase::SArray)
    return fn_ratio(compute_radioactive_heat, rheology, phase)
end

compute_radioactive_heating(rheology, phase::Int) = compute_radioactive_heating(rheology[phase], nothing)

function compute_radioactive_heating(rheology, phase::Nothing)
    if isempty(rheology.RadioactiveHeat)
        return 0.0e0
    else
        compute_phase(compute_radioactive_heat, rheology, phase)
    end
end
