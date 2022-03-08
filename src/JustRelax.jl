module JustRelax

using ParallelStencil
using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using Reexport

include("topology/Topology.jl")
include("MetaJustRelax.jl")
include("stokes/MetaStokes.jl")

@reexport import ParallelStencil

# JustRelax.jl
export PS_Setup, environment!, ps_reset!

end # module
