## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

function Array(
    x::T
) where {T<:Union{StokesArrays,SymmetricTensor,ThermalArrays,Velocity,Residual}}
    return Array(iscpu(x), x)
end

Array(::CPUDeviceTrait, x) = x

function Array(
    ::NonCPUDeviceTrait, x::T
) where {T<:Union{SymmetricTensor,ThermalArrays,Velocity,Residual}}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end

function Array(::NonCPUDeviceTrait, x::StokesArrays{T,A,B,C,M,nDim}) where {T,A,B,C,M,nDim}
    nfields = fieldcount(StokesArrays)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end
