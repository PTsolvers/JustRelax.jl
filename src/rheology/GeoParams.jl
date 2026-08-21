function get_bulk_modulus(args::Vararg{Any, N}) where {N}
    Kb = GeoParams.get_Kb(args...)
    if isnan(Kb) || iszero(Kb)
        return Inf
    end
    return Kb
end

function get_shear_modulus(args::Vararg{Any, N}) where {N}
    G = GeoParams.get_G(args...)
    if isnan(G) || iszero(G)
        return Inf
    end
    return G
end

get_thermal_expansion(args::Vararg{Any, N}) where {N} = get_α(args...)

function get_α(rho::MeltDependent_Density; ϕ::T = 0.0, kwargs...) where {T}
    αsolid = get_α(rho.ρsolid)
    αmelt = get_α(rho.ρmelt)
    return ϕ * αmelt + (1 - ϕ) * αsolid
end

function get_α(rho::BubbleFlow_Density; P = 0.0e0, kwargs...)
    αmelt = get_α(rho.ρmelt)
    αgas = get_α(rho.ρgas)

    @unpack_val c0, a = rho

    cutoff = c0^2 / a^2

    if P < cutoff
        c = a * sqrt(abs(P))
    else
        c = c0
    end

    return inv((c0 - c) / αgas + (1 - (c0 - c)) / αmelt)
end

function get_α(rho::GasPyroclast_Density; kwargs...)
    αmelt = get_α(rho.ρmelt)
    αgas = get_α(rho.ρgas)
    @unpack_val δ, β = rho

    return δ * αgas + (1 - δ) * αmelt
end

function get_α(
        rho::ThreePhase_Density; ϕ = 1.0e0, ϕ_gas = 0.0e0, ϕ_x = (1 - ϕ) * (1 - ϕ_gas), kwargs...
    )
    αmelt = get_α(rho.ρmelt)
    αsolid = get_α(rho.ρsolid)
    αgas = get_α(rho.ρgas; kwargs...)
    ϕ_m = 1 - ϕ_x - ϕ_gas

    return ϕ_m * αmelt + ϕ_gas * αgas + ϕ_x * αsolid
end

@inline function get_α(::Union{RedlichKwong_Density, IdealGas_Density}; T = 0.0e0, kwargs...)
    return iszero(T) ? zero(T) : inv(T)
end

@inline get_α(p::MaterialParams) = get_α(p.Density[1])
@inline get_α(p::MaterialParams, args::NamedTuple) = get_α(p.Density[1], args)
@inline get_α(p::Union{T_Density, PT_Density, Melt_DensityX}) = GeoParams.get_α(p)
@inline get_α(p::Union{T_Density, PT_Density, Melt_DensityX}, ::Any) = GeoParams.get_α(p)
@inline get_α(rho::MeltDependent_Density, args::NamedTuple) = get_α(rho; args...)
@inline get_α(rho::BubbleFlow_Density, args::NamedTuple) = get_α(rho; args...)
@inline get_α(rho::GasPyroclast_Density, args::NamedTuple) = get_α(rho; args...)
@inline get_α(rho::Melt_DensityX, ::Any) = get_α(rho)
@inline get_α(rho::ConstantDensity, args) = 0
@inline get_α(rho::ConstantDensity) = 0
@inline get_α(rho::ThreePhase_Density, args::NamedTuple) = get_α(rho; args...)
@inline get_α(rho::Union{RedlichKwong_Density, IdealGas_Density}, args::NamedTuple) = get_α(rho; args...)
