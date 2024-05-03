import Base: Array

## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

function Array(
    x::T
) where {
    T<:Union{
        JustRelax.StokesArrays,
        JustRelax.SymmetricTensor,
        JustRelax.ThermalArrays,
        JustRelax.Velocity,
        JustRelax.Residual,
        JustRelax.Viscosity,
    },
}
    return Array(backend(x), x)
end

Array(::CPUBackendTrait, x) = x

function Array(
    ::NonCPUBackendTrait, x::T
) where {
    T<:Union{
        JustRelax.SymmetricTensor,
        JustRelax.ThermalArrays,
        JustRelax.Velocity,
        JustRelax.Residual,
        JustRelax.Viscosity,
    },
}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end

function Array(::NonCPUBackendTrait, x::JustRelax.StokesArrays)
    nfields = fieldcount(StokesArrays)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end
