module JustRelax

using Reexport
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using MPI
using GeoParams
using HDF5
using CellArrays
using StaticArrays

function solve!() end

include("types/traits.jl")
export BackendTrait, CPUBackendTrait

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry, velocity_grids, x_g, y_g, z_g

include("phases/CellArrays.jl")
export @cell, element, setelement!, cellnum, cellaxes, new_empty_cell, setindex!

include("JustRelax_CPU.jl")

include("IO/DataIO.jl")

end # module
