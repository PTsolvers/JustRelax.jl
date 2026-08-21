abstract type AbstractMask end

"""
    RockRatio{T, N} <: AbstractMask

Rock-volume fractions on every location of the staggered grid used by the variational Stokes
operators. Fractions lie in `[0, 1]`; zero denotes void and positive values weight the discrete
pressure, velocity, and viscous-stress terms.

The fields are `center`, `vertex`, and the velocity-face arrays `Vx`, `Vy`, `Vz`. In 3D, `yz`,
`xz`, and `xy` store fractions at the corresponding shear-stress locations. Unused 2D fields are
small dummy arrays so that the object remains GPU compatible.

The fraction arrays also determine a reduced linear system. A positive fraction alone does not
always retain a degree of freedom: pressure and velocity unknowns whose equations touch an
eliminated stencil entry are removed according to the validity rules documented by
[`isvalid_c`](@ref), [`isvalid_vx_strict`](@ref), and [`isvalid_vy_strict`](@ref).
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
