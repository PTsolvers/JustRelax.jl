## Velocity type

struct Velocity{T}
    Vx::T
    Vy::T
    Vz::Union{T, Nothing}

    Velocity(Vx::T, Vy::T, Vz::Union{T, Nothing}) where {T} = new{T}(Vx, Vy, Vz)
end

Velocity(Vx::T, Vy::T) where {T} = Velocity(Vx, Vy, nothing)

Velocity(ni::NTuple{N, Number}) where {N} = Velocity(ni...)
function Velocity(::Number, ::Number)
    throw(ArgumentError("Velocity dimensions must be given as integers"))
end
function Velocity(::Number, ::Number, ::Number)
    throw(ArgumentError("Velocity dimensions must be given as integers"))
end

## Displacement type

struct Displacement{T}
    Ux::T
    Uy::T
    Uz::Union{T, Nothing}

    Displacement(Ux::T, Uy::T, Uz::Union{T, Nothing}) where {T} = new{T}(Ux, Uy, Uz)
end

Displacement(Ux::T, Uy::T) where {T} = Displacement(Ux, Uy, nothing)

Displacement(ni::NTuple{N, Number}) where {N} = Displacement(ni...)
function Displacement(::Number, ::Number)
    throw(ArgumentError("Displacement dimensions must be given as integers"))
end
function Displacement(::Number, ::Number, ::Number)
    throw(ArgumentError("Displacement dimensions must be given as integers"))
end

## Vorticity type
struct Vorticity{T}
    yz::Union{T, Nothing}
    xz::Union{T, Nothing}
    xy::T

    function Vorticity(yz::Union{T, Nothing}, xz::Union{T, Nothing}, xy::T) where {T}
        return new{T}(yz, xz, xy)
    end
end

Vorticity(nx::T, ny::T) where {T <: Number} = Vorticity((nx, ny))
Vorticity(nx::T, ny::T, nz::T) where {T <: Number} = Vorticity((nx, ny, nz))
function Vorticity(::NTuple{N, Number}) where {N}
    throw(ArgumentError("Dimensions must be given as integers"))
end

## Viscosity type

struct Viscosity{T}
    η::T # with no plasticity
    η_vep::T # with plasticity
    ητ::T # PT viscosi

    Viscosity(args::Vararg{T, N}) where {T, N} = new{T}(args...)
end

Viscosity(args...) = Viscosity(promote(args...)...)
Viscosity(nx::T, ny::T) where {T <: Number} = Viscosity((nx, ny))
Viscosity(nx::T, ny::T, nz::T) where {T <: Number} = Viscosity((nx, ny, nz))
function Viscosity(::NTuple{N, Number}) where {N}
    throw(ArgumentError("Viscosity dimensions must be given as integers"))
end

## SymmetricTensor type

struct SymmetricTensor{T}
    xx::T
    yy::T
    zz::Union{T, Nothing}
    xy::T
    yz::Union{T, Nothing}
    xz::Union{T, Nothing}
    xy_c::T
    yz_c::Union{T, Nothing}
    xz_c::Union{T, Nothing}
    II::T

    function SymmetricTensor(
            xx::T,
            yy::T,
            zz::Union{T, Nothing},
            xy::T,
            yz::Union{T, Nothing},
            xz::Union{T, Nothing},
            xy_c::T,
            yz_c::Union{T, Nothing},
            xz_c::Union{T, Nothing},
            II::T,
        ) where {T}
        return new{T}(xx, yy, zz, xy, yz, xz, xy_c, yz_c, xz_c, II)
    end
end

function SymmetricTensor(xx::T, yy::T, xy::T, xy_c::T, II::T) where {T}
    return SymmetricTensor(
        xx, yy, nothing, xy, nothing, nothing, xy_c, nothing, nothing, II
    )
end

SymmetricTensor(ni::NTuple{N, Number}) where {N} = SymmetricTensor(ni...)
function SymmetricTensor(::Number, ::Number)
    throw(ArgumentError("SymmetricTensor dimensions must be given as integers"))
end
function SymmetricTensor(::Number, ::Number, ::Number)
    throw(ArgumentError("SymmetricTensor dimensions must be given as integers"))
end

## Residual type

struct Residual{T}
    RP::T
    Rx::T
    Ry::T
    Rz::Union{T, Nothing}

    Residual(RP::T, Rx::T, Ry::T, Rz::Union{T, Nothing}) where {T} = new{T}(RP, Rx, Ry, Rz)
end

Residual(RP::T, Rx::T, Ry::T) where {T} = Residual(RP, Rx, Ry, nothing)
Residual(ni::NTuple{N, Number}) where {N} = Residual(ni...)
function Residual(::Number, ::Number)
    throw(ArgumentError("Residual dimensions must be given as integers"))
end
function Residual(::Number, ::Number, ::Number)
    throw(ArgumentError("Residual dimensions must be given as integers"))
end

## StokesArrays type

struct StokesArrays{A, B, C, D, E, F, T}
    P::T
    P0::T
    V::A
    ∇V::T
    Q::T # volumetric source/sink term
    τ::B
    ε::B
    ε_pl::B
    EII_pl::T
    viscosity::D
    τ_o::Union{B, Nothing}
    R::C
    U::E
    ω::F
    Δε::B
    ∇U::T
end

function StokesArrays(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return StokesArrays(tuple(ni...))
end
StokesArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N} = StokesArrays(ni)
StokesArrays(ni::Vararg{Integer, N}) where {N} = StokesArrays(tuple(ni...))
function StokesArrays(::Number, ::Number)
    throw(ArgumentError("StokesArrays dimensions must be given as integers"))
end
function StokesArrays(::Number, ::Number, ::Number)
    throw(ArgumentError("StokesArrays dimensions must be given as integers"))
end

## PTStokesCoeffs type

struct PTStokesCoeffs{T}
    CFL::T
    ϵ::T # PT tolerance
    Re::T # Reynolds Number
    r::T #
    Vpdτ::T
    θ_dτ::T
    ηdτ::T

    function PTStokesCoeffs(
            li::NTuple{N, T},
            di;
            ϵ::Float64 = 1.0e-8,
            Re::Float64 = 3π,
            CFL::Float64 = (N == 2 ? 0.9 / √2.1 : 0.9 / √3.1),
            r::Float64 = 0.7,
        ) where {N, T}
        lτ = min(li...)
        Vpdτ = min(di...) * CFL
        θ_dτ = lτ * (r + 4 / 3) / (Re * Vpdτ)
        ηdτ = Vpdτ * lτ / Re

        return new{Float64}(CFL, ϵ, Re, r, Vpdτ, θ_dτ, ηdτ)
    end
end
