struct ThermalArrays{_T}
    T::_T  # Temperature @ grid nodes
    Tc::_T # Temperature @ cell centers
    Told::_T
    ΔT::_T
    ΔTc::_T
    dT_dt::_T
    qTx::_T
    qTy::_T
    qTz::Union{_T,Nothing}
    qTx2::_T
    qTy2::_T
    qTz2::Union{_T,Nothing}
    H::_T # source terms
    shear_heating::_T # shear heating terms
    ResT::_T
end

ThermalArrays(::Type{CPUBackend}, ni::NTuple{N,Number}) where {N} = ThermalArrays(ni...)
ThermalArrays(::Type{CPUBackend}, ni::Vararg{Number,N}) where {N} = ThermalArrays(ni...)
ThermalArrays(ni::NTuple{N,Number}) where {N} = ThermalArrays(ni...)
function ThermalArrays(::Number, ::Number)
    throw(ArgumentError("ThermalArrays dimensions must be given as integers"))
end
function ThermalArrays(::Number, ::Number, ::Number)
    throw(ArgumentError("ThermalArrays dimensions must be given as integers"))
end

## Thermal diffusion coefficients

struct PTThermalCoeffs{T,M}
    CFL::T
    ϵ::T
    max_lxyz::T
    max_lxyz2::T
    Vpdτ::T
    θr_dτ::M
    dτ_ρ::M

    function PTThermalCoeffs(
        CFL::T, ϵ::T, max_lxyz::T, max_lxyz2::T, Vpdτ::T, θr_dτ::M, dτ_ρ::M
    ) where {T,M}
        return new{T,M}(CFL, ϵ, max_lxyz, max_lxyz2, Vpdτ, θr_dτ, dτ_ρ)
    end
end
