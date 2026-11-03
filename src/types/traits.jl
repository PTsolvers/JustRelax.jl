"""
    BackendTrait

Supertype for the traits `backend(x)` returns to dispatch solver code on the array type
backing `x`, independent of `x`'s own type hierarchy.
"""
abstract type BackendTrait end
abstract type GPUBackendTrait <: BackendTrait end

"""
    CPUBackendTrait

Trait returned by `backend(x)` when `x` is backed by a plain `Array`.
"""
struct CPUBackendTrait <: BackendTrait end

"""
    NonCPUBackendTrait

Trait returned by `backend(x)` for any `AbstractArray` other than `Array` that isn't a
`CuArray`/`ROCArray` — the generic fallback for GPU array types without a dedicated trait.
"""
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
