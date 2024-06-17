import Base: Array, copy

const JR_T = Union{
    StokesArrays,SymmetricTensor,ThermalArrays,Velocity,Displacement,Residual,Viscosity
}

## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

Array(x::T) where {T<:JR_T} = Array(backend(x), x)
Array(::Nothing) = nothing
Array(::CPUBackendTrait, x) = x

function Array(::GPUBackendTrait, x::T) where {T<:JR_T}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end

function copy(x::T) where {T<:JR_T}
    nfields = fieldcount(T)
    fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        field = getfield(x, i)
        field === nothing ? nothing : copy(field)
    end
    T_clean = remove_parameters(x)
    return T_clean(fields...)
end
