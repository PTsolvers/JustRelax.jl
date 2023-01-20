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
export Geometry, IGG, lazy_grid, init_igg

include("IO/IO.jl")
export save_hdf5

include("MetaJustRelax.jl")
export PS_Setup, environment!, ps_reset!

include("stokes/MetaStokes.jl")

include("thermal_diffusion/MetaDiffusion.jl")

end # module
