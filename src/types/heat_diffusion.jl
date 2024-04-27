struct ThermalArrays{_T}
    T::_T # Temperature @ grid nodes
    Tc::_T # Temperature @ cell centers
    ΔT::_T
    Told::_T
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

    # function ThermalArrays(nx::Integer, ny::Integer)
    #     T = @zeros(nx + 3, ny + 1)
    #     ΔT = @zeros(nx + 3, ny + 1)
    #     Told = @zeros(nx + 3, ny + 1)
    #     Tc = @zeros(nx, ny)
    #     H = @zeros(nx, ny)
    #     shear_heating = @zeros(nx, ny)
    #     dT_dt = @zeros(nx + 1, ny - 1)
    #     qTx = @zeros(nx + 2, ny - 1)
    #     qTy = @zeros(nx + 1, ny)
    #     qTx2 = @zeros(nx + 2, ny - 1)
    #     qTy2 = @zeros(nx + 1, ny)
    #     ResT = @zeros(nx + 1, ny - 1)
    #     return new{typeof(T)}(
    #         T,
    #         Tc,
    #         ΔT,
    #         Told,
    #         dT_dt,
    #         qTx,
    #         qTy,
    #         nothing,
    #         qTx2,
    #         qTy2,
    #         nothing,
    #         H,
    #         shear_heating,
    #         ResT,
    #     )
    # end

    # function ThermalArrays(nx::Integer, ny::Integer, nz::Integer)
    #     T = @zeros(nx + 1, ny + 1, nz + 1)
    #     ΔT = @zeros(nx + 1, ny + 1, nz + 1)
    #     Told = @zeros(nx + 1, ny + 1, nz + 1)
    #     Tc = @zeros(nx, ny, nz)
    #     H = @zeros(nx, ny, nz)
    #     shear_heating = @zeros(nx, ny, nz)
    #     dT_dt = @zeros(ni .- 1)
    #     qTx = @zeros(nx, ny - 1, nz - 1)
    #     qTy = @zeros(nx - 1, ny, nz - 1)
    #     qTz = @zeros(nx - 1, ny - 1, nz)
    #     qTx2 = @zeros(nx, ny - 1, nz - 1)
    #     qTy2 = @zeros(nx - 1, ny, nz - 1)
    #     qTz2 = @zeros(nx - 1, ny - 1, nz)
    #     ResT = @zeros((ni .- 1)...)
    #     return new{typeof(T)}(
    #         T, Tc, ΔT, Told, dT_dt, qTx, qTy, qTz, qTx2, qTy2, qTz2, H, shear_heating, ResT
    #     )
    # end
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

# traits
# @inline backend(::ThermalArrays{T}) where {T} = backend(T)

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

function PTThermalCoeffs(
    K, ρCp, dt, di::NTuple{nDim,T}, li::NTuple{nDim,Any}; ϵ=1e-8, CFL=0.9 / √3
) where {nDim,T}
    Vpdτ = min(di...) * CFL
    max_lxyz = max(li...)
    max_lxyz2 = max_lxyz^2
    Re = @. π + √(π * π + ρCp * max_lxyz2 / K / dt) # Numerical Reynolds number
    θr_dτ = @. max_lxyz / Vpdτ / Re
    dτ_ρ = @. Vpdτ * max_lxyz / K / Re
    return PTThermalCoeffs(CFL, ϵ, max_lxyz, max_lxyz2, Vpdτ, θr_dτ, dτ_ρ)
end
