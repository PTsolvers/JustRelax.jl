import Base: Array, copy
import JustRelax: PTArray, backend

const JR_T = Union{
    StokesArrays,
    DYREL,
    SymmetricTensor,
    ThermalArrays,
    Velocity,
    Displacement,
    Vorticity,
    Residual,
    Viscosity,
}

## Conversion of structs to CPU

@inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

Array(x::T) where {T <: JR_T} = Array(backend(x), x)
Array(::Nothing) = nothing
Array(::CPUBackendTrait, x) = x

function Array(::GPUBackendTrait, x::T) where {T <: JR_T}
    nfields = fieldcount(T)
    cpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        return Array(getfield(x, i))
    end
    T_clean = remove_parameters(x)
    return T_clean(cpu_fields...)
end

## Copy JustRelax custom structs

copy(::Nothing) = nothing

function copy(x::T) where {T <: JR_T}
    nfields = fieldcount(T)
    fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        return field = copy(getfield(x, i))
        # field === nothing ? nothing : copy(field)
    end
    T_clean = remove_parameters(x)
    return T_clean(fields...)
end

# Convert structures to GPU/CPU backends for e.g. restart/checkpointing
function PTArray(backend, x::T) where {T <: JR_T}
    nfields = fieldcount(T)
    gpu_fields = ntuple(Val(nfields)) do i
        Base.@_inline_meta
        field = getfield(x, i)
        return _convert_to_backend(backend, field)
    end
    T_clean = remove_parameters(x)
    return T_clean(gpu_fields...)
end

# Helper function to handle the conversion logic
@inline _convert_to_backend(backend, field::T) where {T <: JR_T} = PTArray(backend, field)
@inline _convert_to_backend(backend, field) = PTArray(backend)(field)
@inline _convert_to_backend(backend, ::Nothing) = nothing
