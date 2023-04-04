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

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry

include("MetaJustRelax.jl")

include("stokes/MetaStokes.jl")
export PS_Setup, environment!, ps_reset!

include("thermal_diffusion/MetaDiffusion.jl")

include("IO/DataIO.jl")

end # module
