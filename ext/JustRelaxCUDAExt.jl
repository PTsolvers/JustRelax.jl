module JustRelaxCUDAExt

using CUDA
using JustPIC
using JustRelax: JustRelax
import JustRelax: PTArray, backend, CUDABackendTrait

PTArray(::Type{CUDABackend}) = CuArray

@inline backend(::CuArray) = CUDABackendTrait()
@inline backend(::Type{<:CuArray}) = CUDABackendTrait()

include("../src/ext/CUDA/2D.jl")
include("../src/ext/CUDA/3D.jl")

end
