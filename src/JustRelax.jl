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

struct CPUBackend end
struct AMDGPUBackend end

PTArray() = Array
PTArray(::Type{CPUBackend}) = Array
PTArray(::T) where {T} = error(ArgumentError("Unknown backend $T"))

export PTArray, CPUBackend, CUDABackend, AMDGPUBackend

include("types/stokes.jl")
# export StokesArrays, PTStokesCoeffs

include("types/heat_diffusion.jl")
# export ThermalArrays, PTThermalCoeffs

include("types/phases.jl")
# export PhaseRatio

include("types/traits.jl")
export BackendTrait, CPUBackendTrait, NonCPUBackendTrait

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry, velocity_grids, x_g, y_g, z_g

include("phases/CellArrays.jl")
export @cell, element, setelement!, cellnum, cellaxes, new_empty_cell, setindex!

include("JustRelax_CPU.jl")

# include("IO/DataIO.jl")

end # module
