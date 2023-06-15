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

include("MiniKernels.jl")
export _d_xa, _d_ya, _d_xi, _d_yi, _av, _av_xa, _av_ya, _gather, _gather_yz, _gather_xz, _gather_xy

include("MetaJustRelax.jl")

include("stokes/MetaStokes.jl")
export PS_Setup, environment!, ps_reset!

include("thermal_diffusion/MetaDiffusion.jl")

include("IO/DataIO.jl")

end # module
