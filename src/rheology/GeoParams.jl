function get_bulk_modulus(args...)
    Kb = GeoParams.get_Kb(args...)
    if isnan(Kb) || iszero(Kb)
        return Inf
    end
    return Kb
end

function get_shear_modulus(args...)
    Kb = GeoParams.get_G(args...)
    if isnan(Kb) || iszero(Kb)
        return Inf
    end
    return Kb
end

function get_thermal_expansion(p)
    α = get_α(p)
    if isnan(α) || iszero(α)
        return 0.0
    end
    return α
end

@inline get_α(p::MaterialParams) = get_α(p.Density[1])
@inline get_α(p::ConstantDensity) = 0.0
@inline get_α(p::Union{T_Density,PT_Density}) = GeoParams.get_α(p)

# Check whether the material has constant density. If so, no need to calculate the density
# during PT iterations
@generated function is_constant_density(
    rheology::NTuple{N,AbstractMaterialParamsStruct}
) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> !_is_constant_density(rheology[i].Density[1]) && return false
        return true
    end
end

@inline _is_constant_density(::ConstantDensity) = true
@inline _is_constant_density(::AbstractDensity) = false

# Check whether the material has a linear viscosity. If so, no need to calculate the viscosity
# during PT iterations
@generated function is_constant_viscosity(
    rheology::NTuple{N,AbstractMaterialParamsStruct}
) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i ->
            is_constant_viscosity(rheology[i].CompositeRheology[1].elements) && return false
        return true
    end
end

@generated function is_constant_viscosity(creep_law::NTuple{N,AbstractCreepLaw}) where {N}
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> _is_constant_viscosity(creep_law[i]) && return false
        return true
    end
end

@inline _is_constant_viscosity(::Union{LinearViscous,ConstantElasticity}) = true
@inline _is_constant_viscosity(::AbstractCreepLaw) = false
