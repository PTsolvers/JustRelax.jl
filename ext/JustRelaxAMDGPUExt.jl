module JustRelaxAMDGPUExt

using AMDGPU
using JustPIC
using JustRelax: JustRelax
import JustRelax: PTArray, backend, AMDGPUBackendTrait, AMDGPUBackend

PTArray(::Type{AMDGPUBackend}) = ROCArray

@inline backend(::ROCArray) = AMDGPUBackendTrait()
@inline backend(::Type{<:ROCArray}) = AMDGPUBackendTrait()

include("../src/ext/AMDGPU/2D.jl")
include("../src/ext/AMDGPU/3D.jl")

end
