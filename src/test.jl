import Pkg; Pkg.activate(".")
using ParallelStencil
using JustRelax
using GeoParams

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


model = PS_Setup(:cpu, Float64, 2)
environment!(model)

# Geometry
ni = (256-1, 256-1)
li = (10.0, 10.0)  # domain extends

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
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct SymmetricTensor{T} \n"*
        join("$(dims[i])$(dims[j])::T\n" for i in 1:nDim, j in 1:nDim if j≥i)*
        "end"
    )
    eval(str)
end
