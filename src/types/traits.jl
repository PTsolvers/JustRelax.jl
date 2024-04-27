abstract type BackendTrait end
struct CPUBackendTrait <: BackendTrait end
struct NonCPUBackendTrait <: BackendTrait end
# struct CUDABackendTrait <: BackendTrait end
# struct AMDGPUBackendTrait <: BackendTrait end

@inline backend(::Array) = CPUBackendTrait()
@inline backend(::Type{<:Array}) = CPUBackendTrait()
@inline backend(::AbstractArray) = NonCPUDeviceTrait()
@inline backend(::Type{<:AbstractArray}) = NonCPUDeviceTrait()

@inline backend(::T) where {T} = throw(ArgumentError("Backend $(T) not supported"))

@inline backend(::JustRelax.Velocity{T}) where {T} = backend(T)
@inline backend(::JustRelax.SymmetricTensor{T}) where {T} = backend(T)
@inline backend(::JustRelax.Residual{T}) where {T} = backend(T)
@inline backend(::JustRelax.ThermalArrays{T}) where {T} = backend(T)
@inline backend(x::JustRelax.StokesArrays) = backend(x.P)
@inline backend(x::JustRelax.PhaseRatio) = backend(x.center.data)
