module JustRelax

using ParallelStencil
using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA

include("topology/Topology.jl")
include("MetaJustRelax.jl")
include("stokes/MetaStokes.jl")

# ParallelStencil.jl exports
import ParallelStencil: @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand
export @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand

# JustRelax.jl
export PS_Setup, environment!, ps_reset!

end # module
