import Base: Array

## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

const JR_T = Union{
    JustRelax.StokesArrays,
    JustRelax.SymmetricTensor,
    JustRelax.ThermalArrays,
    JustRelax.Velocity,
    JustRelax.Residual,
}

Array(::CPUBackendTrait, x) = x
Array(x::T) where {T<:JR_T} = Array(backend(x), x)

function Array(::GPUBackendTrait, x::T) where {T<:JR_T}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end
