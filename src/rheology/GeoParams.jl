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

function get_α(rho::MeltDependent_Density; ϕ::T=0.0, kwargs...) where {T}
    αsolid = get_α(rho.ρsolid)
    αmelt  = get_α(rho.ρmelt)
    return ϕ * αmelt + (1-ϕ) * αsolid
end

@inline get_α(p::MaterialParams) = get_α(p.Density[1])
@inline get_α(p::MaterialParams, args::NamedTuple) = get_α(p.Density[1], args)
@inline get_α(p::Union{T_Density,PT_Density}) = GeoParams.get_α(p)
@inline get_α(rho::MeltDependent_Density, args) = get_α(rho; args...)
@inline get_α(rho::ConstantDensity, args) = 0
@inline get_α(rho::ConstantDensity) = 0
