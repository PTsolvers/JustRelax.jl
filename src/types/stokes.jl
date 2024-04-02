# abstract type AbstractStokesModel end
# abstract type AbstractViscosity end
# abstract type Viscous <: AbstractStokesModel end
# abstract type AbstractElasticModel <: AbstractStokesModel end
# abstract type ViscoElastic <: AbstractElasticModel end
# abstract type ViscoElastoPlastic <: AbstractElasticModel end

## Velocity type

struct Velocity{T}
    Vx::T
    Vy::T
    Vz::Union{T,Nothing}

    Velocity(Vx::T, Vy::T, Vz::Union{T,Nothing}) where {T} = new{T}(Vx, Vy, Vz)
end

Velocity(Vx::T, Vy::T) where {T} = Velocity(Vx, Vy, nothing)

function Velocity(nx::Integer, ny::Integer)
    nVx = (nx + 1, ny + 2)
    nVy = (nx + 2, ny + 1)

    Vx, Vy = @zeros(nVx...), @zeros(nVy)
    return Velocity(Vx, Vy, nothing)
end

function Velocity(nx::Integer, ny::Integer, nz::Integer)
    nVx = (nx + 1, ny + 2, nz + 2)
    nVy = (nx + 2, ny + 1, nz + 2)
    nVz = (nx + 2, ny + 2, nz + 1)

    Vx, Vy, Vz = @zeros(nVx...), @zeros(nVy), @zeros(nVz)
    return Velocity{PTArray}(Vx, Vy, Vz)
end

Velocity(ni::NTuple{N,Number}) where {N} = Velocity(ni...)
function Velocity(::Number, ::Number)
    throw(ArgumentError("Velocity dimensions must be given as integers"))
end
function Velocity(::Number, ::Number, ::Number)
    throw(ArgumentError("Velocity dimensions must be given as integers"))
end

## Viscosity type

struct Viscosity{T}
    η::T # with no plasticity
    η_vep::T # with plasticity
    ητ::T # PT viscosi

    Viscosity(args::Vararg{T,N}) where {T,N} = new{T}(args...)
end

Viscosity(args...) = Viscosity(promote(args...)...)

function Viscosity(ni::NTuple{N,Integer}) where {N}
    η = @ones(ni...)
    η_vep = @ones(ni...)
    ητ = @zeros(ni...)
    return Viscosity(η, η_vep, ητ)
end

Viscosity(nx::T, ny::T) where {T<:Number} = Viscosity((nx, ny))
Viscosity(nx::T, ny::T, nz::T) where {T<:Number} = Viscosity((nx, ny, nz))
function Viscosity(::NTuple{N,Number}) where {N}
    throw(ArgumentError("Viscosity dimensions must be given as integers"))
end

## SymmetricTensor type

struct SymmetricTensor{T}
    xx::T
    yy::T
    zz::Union{T,Nothing}
    xy::T
    yz::Union{T,Nothing}
    xz::Union{T,Nothing}
    xy_c::T
    yz_c::Union{T,Nothing}
    xz_c::Union{T,Nothing}
    II::T

    function SymmetricTensor(
        xx::T,
        yy::T,
        zz::Union{T,Nothing},
        xy::T,
        yz::Union{T,Nothing},
        xz::Union{T,Nothing},
        xy_c::T,
        yz_c::Union{T,Nothing},
        xz_c::Union{T,Nothing},
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

function SymmetricTensor(nx::Integer, ny::Integer)
    return SymmetricTensor(
        @zeros(nx, ny), # xx
        @zeros(nx, ny), # yy
        @zeros(nx + 1, ny + 1), # xy
        @zeros(nx, ny), # xy @ cell center
        @zeros(nx, ny) # II (second invariant)
    )
end

function SymmetricTensor(nx::Integer, ny::Integer, nz::Integer)
    return SymmetricTensor(
        @zeros(nx, ny, nz), # xx
        @zeros(nx + 1, ny + 1, nz), # xy
        @zeros(nx, ny, nz), # yy
        @zeros(nx + 1, ny, nz + 1), # xz
        @zeros(nx, ny + 1, nz + 1), # yz
        @zeros(nx, ny, nz), # zz
        @zeros(nx, ny, nz), # yz @ cell center
        @zeros(nx, ny, nz), # xz @ cell center
        @zeros(nx, ny, nz), # xy @ cell center
        @zeros(nx, ny, nz), # II (second invariant)
    )
end

SymmetricTensor(ni::NTuple{N,Number}) where {N} = SymmetricTensor(ni...)
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
    Rz::Union{T,Nothing}

    Residual(RP::T, Rx::T, Ry::T, Rz::Union{T,Nothing}) where {T} = new{T}(RP, Rx, Ry, Rz)
end

Residual(RP::T, Rx::T, Ry::T) where {T} = Residual(RP, Rx, Ry, nothing)

function Residual(nx::Integer, ny::Integer)
    Rx = @zeros(nx - 1, ny)
    Ry = @zeros(nx, ny - 1)
    RP = @zeros(nx, ny)
    return Residual(RP, Rx, Ry)
end

function Residual(nx::Integer, ny::Integer, nz::Integer)
    Rx = @zeros(nx - 1, ny, nz)
    Ry = @zeros(nx, ny - 1, nz)
    Rz = @zeros(nx, ny, nz - 1)
    RP = @zeros(nx, ny, nz)
    return Residual(RP, Rx, Ry, Rz)
end

Residual(ni::NTuple{N,Number}) where {N} = Residual(ni...)
function Residual(::Number, ::Number)
    throw(ArgumentError("Residual dimensions must be given as integers"))
end
function Residual(::Number, ::Number, ::Number)
    throw(ArgumentError("Residual dimensions must be given as integers"))
end

## StokesArrays type

struct StokesArrays{A,B,C,D,T}
    P::T
    P0::T
    V::A
    ∇V::T
    τ::B
    ε::B
    ε_pl::B
    EII_pl::T
    viscosity::D
    τ_o::Union{B,Nothing}
    R::C

    function StokesArrays(ni::NTuple{N,Integer}) where {N}
        P = @zeros(ni...)
        P0 = @zeros(ni...)
        ∇V = @zeros(ni...)
        V = Velocity(ni)
        τ = SymmetricTensor(ni)
        τ_o = SymmetricTensor(ni)
        ε = SymmetricTensor(ni)
        ε_pl = SymmetricTensor(ni)
        EII_pl = @zeros(ni...)
        viscosity = Viscosity(ni)
        R = Residual(ni)

        return new{typeof(V),typeof(τ),typeof(R),typeof(viscosity),typeof(P)}(
            P, P0, V, ∇V, τ, ε, ε_pl, EII_pl, viscosity, τ_o, R
        )
    end
end

StokesArrays(ni::Vararg{Integer,N}) where {N} = StokesArrays(tuple(ni...))
function StokesArrays(::Number, ::Number)
    throw(ArgumentError("StokesArrays dimensions must be given as integers"))
end
function StokesArrays(::Number, ::Number, ::Number)
    throw(ArgumentError("StokesArrays dimensions must be given as integers"))
end

# traits
@inline backend(x::StokesArrays) = backend(x.P)

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
        li::NTuple{N,T},
        di;
        ϵ::Float64=1e-8,
        Re::Float64=3π,
        CFL::Float64=(N == 2 ? 0.9 / √2.1 : 0.9 / √3.1),
        r::Float64=0.7,
    ) where {N,T}
        lτ = min(li...)
        Vpdτ = min(di...) * CFL
        θ_dτ = lτ * (r + 4 / 3) / (Re * Vpdτ)
        ηdτ = Vpdτ * lτ / Re

        return new{Float64}(CFL, ϵ, Re, r, Vpdτ, θ_dτ, ηdτ)
    end
end
