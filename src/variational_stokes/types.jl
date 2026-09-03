abstract type AbstractMask end

"""
    RockRatio{T, N} <: AbstractMask

A struct representing the rock ratio mask used in variational Stokes solvers. It contains arrays for center and vertex values, as well as velocity components and shear components.
"""
struct RockRatio{T, N} <: AbstractMask
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::T
    yz::T
    xz::T
    xy::T

    function RockRatio(
            center::AbstractArray{F, N}, vertex::T, Vx::T, Vy::T, Vz::T, yz::T, xz::T, xy::T
        ) where {F, N, T}
        return new{T, N}(center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end
end

RockRatio(::Type{CPUBackend}, ni::NTuple{N, Number}) where {N} = RockRatio(ni...)

function RockRatio(::Number, ::Number)
    throw(ArgumentError("RockRatio dimensions must be given as integers"))
end

function RockRatio(::Number, ::Number, ::Number)
    throw(ArgumentError("RockRatio dimensions must be given as integers"))
end

Adapt.@adapt_structure RockRatio
