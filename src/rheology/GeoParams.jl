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
@inline get_α(p::Union{T_Density, PT_Density}) = GeoParams.get_α(p)