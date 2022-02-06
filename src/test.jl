import Pkg; Pkg.activate(".")
using ParallelStencil, JustRelax
using GeoParams

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

@zeros(2,2)

# Physics
ni = (256-1, 256-1)
li = (10.0, 10.0)  # domain extends
# # nx, ny    = 1*128-1, 1*128-1    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
# # Derived numerics
# dx, dy    = lx/nx, ly/ny # cell sizes
# max_lxy   = max(lx,ly)
# xc, yc, yv = LinRange(dx/2, lx - dx/2, nx), LinRange(dy/2, ly - dy/2, ny), LinRange(0, ly, ny+1)

struct Geometry{nDim}
    ni::NTuple
    li::NTuple
    max_li::Float64
    di::NTuple
    xci::NTuple
    xvi::NTuple

    function Geometry(ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}) where {nDim, T}
        di = li./ni
        new{nDim}(
            ni,
            li,
            Float64(max(li...)),
            di,
            Tuple([di[i]/2:di[i]:li[i]-di[i]/2 for i in 1:nDim]),
            Tuple([0:(li[i]/(ni[i]+1)):li[i] for i in 1:nDim])
        )
    end

end

abstract type AbstractVelocity end

struct Velocity1D{T} <: AbstractVelocity
    Vx::T
end

struct Velocity2D{T} <: AbstractVelocity
    Vx::T
    Vy::T
end

struct Velocity3D{T} <: AbstractVelocity
    Vx::T
    Vy::T
    Vz::T
end

struct StokesArrays

end

geometry = Geometry(ni, li)

function make_velocity_struct(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct Velocity{T} \n"*
        join("V$(dims[i])::T\n" for i in 1:nDim)*
        "end"
    )
    eval(str)
end

function make_tensor_struct(nDim::Integer)
    dims = [
        "xx" "xy" "xz"
        "yx" "yy" "yz"
        "zx" "zy" "zz"
    ]
    str = Meta.parse(
        "struct SymmetricTensor{T} \n"*
        join("$(dims[i,j])::T\n" for i in 1:nDim, j in 1:nDim if j≥i)*
        "end"
    )
    eval(str)
end
