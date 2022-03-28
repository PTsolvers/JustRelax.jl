module JustRelax

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI

include("topology/Topology.jl")
include("MetaJustRelax.jl")
include("stokes/MetaStokes.jl")

export PS_Setup, environment!, ps_reset!

end # module
