module JustRelax

using Adapt
using Reexport
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using MPI
using GeoParams
using HDF5
using CellArrays
using StaticArrays
using Statistics
@reexport using JustPIC

function solve!() end

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct AMDGPUBackend <: AbstractBackend end

PTArray() = Array
PTArray(::Type{CPUBackend}) = Array
PTArray(::T) where {T} = error(ArgumentError("Unknown backend $T"))

export PTArray, CPUBackend, CUDABackend, AMDGPUBackend

include("types/stokes.jl")

include("types/heat_diffusion.jl")

include("variational_stokes/types.jl")

include("types/weno.jl")

include("boundaryconditions/types.jl")
export TemperatureBoundaryConditions,
    DisplacementBoundaryConditions, VelocityBoundaryConditions

include("types/traits.jl")
export BackendTrait, CPUBackendTrait, NonCPUBackendTrait

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry, velocity_grids, x_g, y_g, z_g

include("JustRelax_CPU.jl")

include("IO/DataIO.jl")

include("types/type_conversions.jl")
export Array, copy

end # module
