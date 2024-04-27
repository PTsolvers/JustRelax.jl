# Device trait system 

abstract type DeviceTrait end
struct CPUDeviceTrait <: DeviceTrait end
struct NonCPUDeviceTrait <: DeviceTrait end

@inline iscpu(::Array) = CPUDeviceTrait()
@inline iscpu(::AbstractArray) = NonCPUDeviceTrait()
@inline iscpu(::T) where T = throw(ArgumentError("Unkown device"))

@inline iscpu(::Velocity{Array{T, N}}) where {T, N} = CPUDeviceTrait()
@inline iscpu(::Velocity{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

@inline iscpu(::SymmetricTensor{Array{T, N}}) where {T, N} = CPUDeviceTrait()
@inline iscpu(::SymmetricTensor{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

@inline iscpu(::Residual{Array{T, N}}) where {T, N} = CPUDeviceTrait()
@inline iscpu(::Residual{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

@inline iscpu(::ThermalArrays{Array{T, N}}) where {T, N} = CPUDeviceTrait()
@inline iscpu(::ThermalArrays{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

@inline iscpu(::StokesArrays{M,A,B,C,Array{T, N},nDim}) where {M,A,B,C,T,N,nDim} =CPUDeviceTrait()
@inline iscpu(::StokesArrays{M,A,B,C,AbstractArray{T, N},nDim}) where {M,A,B,C,T,N,nDim} =NonCPUDeviceTrait()

## Conversion of structs to CPU

@inline remove_parameters(::T) where T = Base.typename(T).wrapper

function Array(x::T) where T<:Union{StokesArrays, SymmetricTensor, ThermalArrays, Velocity, Residual}
    Array(iscpu(x), x)
end

Array(::CPUDeviceTrait, x) = x

function Array(::NonCPUDeviceTrait, x::T) where T<:Union{SymmetricTensor, ThermalArrays, Velocity, Residual}
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
