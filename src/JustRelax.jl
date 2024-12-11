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
using Statistics
@reexport using JustPIC

function solve!() end
function __init__()
    println("""
         _           _   ____      _               _ _
        | |_   _ ___| |_|  _ \\ ___| | __ ___  __  (_) |
     _  | | | | / __| __| |_) / _ \\ |/ _` \\ \\/ /  | | |
    | |_| | |_| \\__ \\ |_|  _ <  __/ | (_| |>  < _ | | |
     \\___/ \\__,_|___/\\__|_| \\_\\___|_|\\__,_/_/\\_(_)/ |_|
                                                |__/
    """)
end

abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct AMDGPUBackend <: AbstractBackend end

PTArray() = Array
PTArray(::Type{CPUBackend}) = Array
PTArray(::T) where {T} = error(ArgumentError("Unknown backend $T"))

export PTArray, CPUBackend, CUDABackend, AMDGPUBackend

include("stress_rotation/types.jl")
export unwrap

include("types/stokes.jl")
# export StokesArrays, PTStokesCoeffs

include("types/heat_diffusion.jl")
# export ThermalArrays, PTThermalCoeffs

include("types/weno.jl")

include("mask/mask.jl")

include("boundaryconditions/Dirichlet.jl")

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
