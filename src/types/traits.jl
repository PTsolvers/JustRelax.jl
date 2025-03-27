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

for type in (
        JustRelax.Velocity,
        JustRelax.Displacement,
        JustRelax.Vorticity,
        JustRelax.SymmetricTensor,
        JustRelax.Residual,
        JustRelax.Viscosity,
        JustRelax.ThermalArrays,
    )
    @eval @inline backend(::$(type){T}) where {T} = backend(T)
end

@inline backend(x::JustRelax.StokesArrays) = backend(x.P)
# @inline backend(x::JustPIC.PhaseRatios) = backend(x.center.data)

# Error handling
@inline backend(::T) where {T} = throw(ArgumentError("$(T) is not a supported backend"))
