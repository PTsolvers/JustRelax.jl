abstract type BackendTrait end
struct CPUBackendTrait <: BackendTrait end
# struct CUDABackendTrait <: BackendTrait end
# struct AMDGPUBackendTrait <: BackendTrait end

@inline backend(::Array) = CPUBackendTrait()
@inline backend(::Type{<:Array}) = CPUBackendTrait()
@inline backend(::T) where {T} = throw(ArgumentError("Backend $(T) not supported"))
