abstract type BackendTrait end
abstract type GPUBackendTrait <: BackendTrait end

struct CPUBackendTrait <: BackendTrait end
struct NonCPUBackendTrait <: GPUBackendTrait end
struct CUDABackendTrait <: GPUBackendTrait end
struct AMDGPUBackendTrait <: GPUBackendTrait end

# AbstractArray's
@inline backend(::Array) = CPUBackendTrait()
@inline backend(::Type{<:Array}) = CPUBackendTrait()
@inline backend(::AbstractArray) = NonCPUBackendTrait()
@inline backend(::Type{<:AbstractArray}) = NonCPUBackendTrait()

# Custom struct's
@inline backend(::JustRelax.Velocity{T}) where {T} = backend(T)
@inline backend(::JustRelax.SymmetricTensor{T}) where {T} = backend(T)
@inline backend(::JustRelax.Residual{T}) where {T} = backend(T)
@inline backend(::JustRelax.Viscosity{T}) where {T} = backend(T)
@inline backend(::JustRelax.ThermalArrays{T}) where {T} = backend(T)
@inline backend(x::JustRelax.StokesArrays) = backend(x.P)
@inline backend(x::JustRelax.PhaseRatio) = backend(x.center.data)

# Error handling
@inline backend(::T) where {T} = throw(ArgumentError("$(T) is not a supported backend"))
