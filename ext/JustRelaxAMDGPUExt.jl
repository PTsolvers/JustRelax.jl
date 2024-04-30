module JustRelaxAMDGPUExt

using AMDGPU
using JustRelax: JustRelax
import JustRelax: PTArray, backend, AMDGPUBackendTrait

PTArray(::Type{AMDGPUBackend}) = RocArray

@inline backend(::CuArray) = AMDGPUBackendTrait()
@inline backend(::Type{<:CuArray}) = AMDGPUBackendTrait()

include("../src/ext/AMDGPU/2D.jl")
include("../src/ext/AMDGPU/3D.jl")

end
