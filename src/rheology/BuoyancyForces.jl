"""
    compute_ρg!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""
function compute_ρg!(ρg, rheology, args)
    _size(x::AbstractArray) = size(x)
    _size(x::NTuple) = size(x[1])

    ni = _size(ρg)
    @parallel (@idx ni) compute_ρg_kernel!(ρg, rheology, args)
    return nothing
end

@parallel_indices (I...) function compute_ρg_kernel!(ρg, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    @inbounds ρg[I...] = compute_buoyancy(rheology, args_ijk)
    return nothing
end

@parallel_indices (I...) function compute_ρg_kernel!(
        ρg::NTuple{N, AbstractArray}, rheology, args
    ) where {N}
    args_ijk = ntuple_idx(args, I...)
    gᵢ = compute_gravity(first(rheology))
    ρgᵢ = compute_buoyancies(rheology, args_ijk, gᵢ, Val(N))
    fill_density!(ρg, ρgᵢ, I...)
    return nothing
end

"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
The `phase_ratios` are used to compute the density of the composite rheology.
"""
function compute_ρg!(ρg, phase_ratios::JustPIC.PhaseRatios, rheology, args)
    _size(x::AbstractArray) = size(x)
    _size(x::NTuple) = size(x[1])

    ni = _size(ρg)
    @parallel (@idx ni) compute_ρg_kernel!(ρg, phase_ratios.center, rheology, args)
    return nothing
end

@parallel_indices (I...) function compute_ρg_kernel!(ρg, phase_ratios, rheology, args)
    args_ijk = ntuple_idx(args, I...)
    ρg[I...] = compute_buoyancy(rheology, args_ijk, @cell(phase_ratios[I.+1...]))
    return nothing
end

@parallel_indices (I...) function compute_ρg_kernel!(
        ρg::NTuple{N, AbstractArray}, phase_ratios, rheology, args
    ) where {N}
    args_ijk = ntuple_idx(args, I...)
    gᵢ = compute_gravity(first(rheology))
    ρgᵢ = compute_buoyancies(rheology, @cell(phase_ratios[I.+1...]), args_ijk, gᵢ, Val(N))
    fill_density!(ρg, ρgᵢ, I...)
    return nothing
end

## Inner buoyancy force kernels
@generated function fill_density!(ρg::NTuple{N}, ρgᵢ::NTuple{N}, I::Vararg{Int, N}) where {N}
    return quote
        Base.@nexprs $N i -> @inbounds ρg[i][I...] = ρgᵢ[i]
        return nothing
    end
end

@inline fill_density!(ρg::NTuple{N}, ρgᵢ::Number, I::Vararg{Int, N}) where {N} =
    setindex!(last(ρg), ρgᵢ, I...)

@inline compute_buoyancies(rheology::MaterialParams, args, gᵢ::NTuple{3}, ::Val{2}) =
    compute_density(rheology, args) .* (gᵢ[1], gᵢ[3])
@inline compute_buoyancies(rheology::MaterialParams, args, gᵢ::NTuple{3}, ::Val{3}) =
    compute_density(rheology, args) .* gᵢ
@inline compute_buoyancies(rheology::MaterialParams, args, gᵢ::Number, ::Any) =
    compute_density(rheology, args) * gᵢ

@inline compute_buoyancies(
    rheology::MaterialParams, phase_ratios, args, gᵢ::NTuple{3}, ::Val{2}
) = compute_density_ratio(phase_ratios, rheology, args) .* (gᵢ[1], gᵢ[3])
@inline compute_buoyancies(
    rheology::MaterialParams, phase_ratios, args, gᵢ::NTuple{3}, ::Val{3}
) = compute_density_ratio(phase_ratios, rheology, args) .* gᵢ
@inline compute_buoyancies(
    rheology::MaterialParams, phase_ratios, args, gᵢ::Number, ::Any
) = compute_density_ratio(phase_ratios, rheology, args) * gᵢ

@inline compute_buoyancies(rheology, phase_ratios, args, gᵢ::NTuple{3}, ::Val{2}) =
    fn_ratio(compute_density, rheology, phase_ratios, args) .* (gᵢ[1], gᵢ[3])
@inline compute_buoyancies(rheology, phase_ratios, args, gᵢ::NTuple{3}, ::Val{3}) =
    fn_ratio(compute_density, rheology, phase_ratios, args) .* gᵢ
@inline compute_buoyancies(rheology, phase_ratios, args, gᵢ::Number, ::Any) =
    fn_ratio(compute_density, rheology, phase_ratios, args) * gᵢ

"""
    compute_buoyancy(rheology::MaterialParams, args)

Compute the buoyancy forces based on the given rheology parameters and arguments.

# Arguments
- `rheology::MaterialParams`: The material parameters for the rheology.
- `args`: The arguments for the computation.
"""
@inline function compute_buoyancy(rheology::MaterialParams, args)
    return compute_density(rheology, args) * compute_gravity(rheology)
end

"""
    compute_buoyancy(rheology::MaterialParams, args, phase_ratios)

Compute the buoyancy forces for a given set of material parameters, arguments, and phase ratios.

# Arguments
- `rheology`: The material parameters.
- `args`: The arguments.
- `phase_ratios`: The phase ratios.
"""
@inline function compute_buoyancy(rheology::MaterialParams, args, phase_ratios)
    return compute_density_ratio(phase_ratios, rheology, args) * compute_gravity(rheology)
end

"""
    compute_buoyancy(rheology, args)

Compute the buoyancy forces based on the given rheology and arguments.

# Arguments
- `rheology`: The rheology used to compute the buoyancy forces.
- `args`: Additional arguments required for the computation.
"""
@inline function compute_buoyancy(rheology, args)
    return compute_density(rheology, args) * compute_gravity(rheology[1])
end

"""
    compute_buoyancy(rheology, args, phase_ratios)

Compute the buoyancy forces based on the given rheology, arguments, and phase ratios.

# Arguments
- `rheology`: The rheology used to compute the buoyancy forces.
- `args`: Additional arguments required by the rheology.
- `phase_ratios`: The ratios of the different phases.
"""
@inline function compute_buoyancy(rheology, args, phase_ratios)
    return fn_ratio(compute_density, rheology, phase_ratios, args) *
        compute_gravity(rheology[1])
end

# without phase ratios
@inline update_ρg!(ρg::Union{NTuple, AbstractArray}, rheology, args) =
    update_ρg!(isconstant(rheology), ρg, rheology, args)
@inline update_ρg!(::ConstantDensityTrait, ρg, rheology, args) = nothing
@inline update_ρg!(::NonConstantDensityTrait, ρg, rheology, args) =
    compute_ρg!(ρg, rheology, args)
# with phase ratios
@inline update_ρg!(
    ρg::Union{NTuple, AbstractArray}, phase_ratios::JustPIC.PhaseRatios, rheology, args
) = update_ρg!(isconstant(rheology), ρg, phase_ratios, rheology, args)
@inline update_ρg!(
    ::ConstantDensityTrait, ρg, phase_ratios::JustPIC.PhaseRatios, rheology, args
) = nothing
@inline update_ρg!(
    ::NonConstantDensityTrait, ρg, phase_ratios::JustPIC.PhaseRatios, rheology, args
) = compute_ρg!(ρg, phase_ratios, rheology, args)
