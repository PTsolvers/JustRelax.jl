abstract type AbstractMask end

struct RockRatio{T,N} <: AbstractMask
    center::T
    vertex::T
    Vx::T
    Vy::T
    Vz::Union{Nothing,T}
    yz::Union{Nothing,T}
    xz::Union{Nothing,T}
    xy::Union{Nothing,T}

    function RockRatio(
        center::AbstractArray{F,N},
        vertex::T,
        Vx::T,
        Vy::T,
        Vz::Union{Nothing,T},
        yz::Union{Nothing,T},
        xz::Union{Nothing,T},
        xy::Union{Nothing,T},
    ) where {F,N,T}
        return new{T,N}(center, vertex, Vx, Vy, Vz, yz, xz, xy)
    end
end

RockRatio(ni::NTuple{N,Integer}) where {N} = RockRatio(ni...)

Adapt.@adapt_structure RockRatio
