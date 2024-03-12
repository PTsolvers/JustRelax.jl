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

function get_thermal_expansion(args...)
    α = GeoParams.get_α(args...)
    if isnan(α) || iszero(α)
        return 0.0
    end
    return α
end
