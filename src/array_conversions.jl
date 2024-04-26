
import Base.Array

@inline remove_parameters(::T) where T = Base.typename(T).wrapper

function Array(x::T) where T<:Union{SymmetricTensor, ThermalArrays, Velocity, Residual}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end

function Array(x::StokesArrays{T,A,B,C,M,nDim}) where {T,A,B,C,M,nDim}
    nfields = fieldcount(StokesArrays)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end
