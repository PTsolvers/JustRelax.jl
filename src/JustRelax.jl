module JustRelax

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using CUDA
using MPI
using GeoParams
using HDF5

function solve!() end

function solve!() end

include("topology/Topology.jl")
include("MetaJustRelax.jl")
include("stokes/MetaStokes.jl")
include("thermal_diffusion/MetaDiffusion.jl")
include("IO/DataIO.jl")

include("MetaJustRelax.jl")
export PS_Setup, environment!, ps_reset!

include("stokes/MetaStokes.jl")

include("thermal_diffusion/MetaDiffusion.jl")

end # module
