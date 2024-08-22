function get_bulk_modulus(args::Vararg{Any,N}) where {N}
    Kb = GeoParams.get_Kb(args...)
    if isnan(Kb) || iszero(Kb)
        return Inf
    end
    return Kb
end

function get_shear_modulus(args::Vararg{Any,N}) where {N}
    Kb = GeoParams.get_G(args...)
    if isnan(Kb) || iszero(Kb)
        return Inf
    end
    return Kb
end

get_thermal_expansion(args::Vararg{Any,N}) where {N} = get_α(args...)

@inline get_α(p::MaterialParams) = get_α(p.Density[1])
@inline get_α(p::ConstantDensity) = 0.0
@inline get_α(p::Union{T_Density,PT_Density}) = GeoParams.get_α(p)
@inline get_α(p::MeltDependent_Density, ϕ) = GeoParams.get_α(p, (; ϕ = ϕ))
