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
using TOML, Crayons
@reexport using JustPIC

function solve!() end
#! format: off
function __init__(io = stdout)
    j = string(Crayon(foreground = (50,74,201)))
    u = string(Crayon(foreground = (50,74,201)))
    s = string(Crayon(foreground = (50,74,201)))
    t = string(Crayon(foreground = (50,74,201)))
    r = string(Crayon(foreground = (47,138,29)))
    e = string(Crayon(foreground = (47,138,29)))
    l = string(Crayon(foreground = (189,39,39)))
    a = string(Crayon(foreground = (189,39,39)))
    x = string(Crayon(foreground = (130,63,163)))
    res = string(Crayon(reset = true))

    str = """
     $(j)██╗$(u)██╗   ██╗$(s)███████╗$(t)████████╗$(r)██████╗ $(e)███████╗$(l)██╗      $(a)█████╗ $(x)██╗  ██╗$(res)
     $(j)██║$(u)██║   ██║$(s)██╔════╝$(t)╚══██╔══╝$(r)██╔══██╗$(e)██╔════╝$(l)██║     $(a)██╔══██╗$(x)╚██╗██╔╝$(res)
     $(j)██║$(u)██║   ██║$(s)███████╗$(t)   ██║   $(r)██████╔╝$(e)█████╗  $(l)██║     $(a)███████║$(x) ╚███╔╝$(res)
$(j)██   ██║$(u)██║   ██║$(s)╚════██║$(t)   ██║   $(r)██╔══██╗$(e)██╔══╝  $(l)██║     $(a)██╔══██║$(x) ██╔██╗$(res)
$(j)╚█████╔╝$(u)╚██████╔╝$(s)███████║$(t)   ██║   $(r)██║  ██║$(e)███████╗$(l)███████╗$(a)██║  ██║$(x)██╔╝ ██╗$(res)
 $(j)╚════╝ $(u) ╚═════╝ $(s)╚══════╝$(t)   ╚═╝   $(r)╚═╝  ╚═╝$(e)╚══════╝$(l)╚══════╝$(a)╚═╝  ╚═╝$(x)╚═╝  ╚═╝$(res)
     """
    printstyled(io, "\n\n", str, "\n",
"""
Version: $(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
Latest commit: $(try strip(read(`git log -1 --pretty=%B`, String)) catch _ "N/A" end)
Commit date: $(try strip(read(`git log -1 --pretty=%cd`, String)) catch _ "N/A" end)
""", bold=true, color=:default)
end
#! format: on
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

include("types/heat_diffusion.jl")

include("variational_stokes/types.jl")

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
