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

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry, velocity_grids, x_g, y_g, z_g

# include("MiniKernels.jl")

include("phases/CellArrays.jl")
export @cell, element, setelement!, cellnum, cellaxes, new_empty_cell, setindex!

# include("rheology/StressUpdate.jl")
# export plastic_params, plastic_params_phase, compute_dτ_r, _compute_τ_nonlinear!

include("JustRelax_CPU.jl")
# include("MetaJustRelax.jl")

# include("stokes/MetaStokes.jl")
# export PS_Setup, environment!, ps_reset!

# include("thermal_diffusion/MetaDiffusion.jl")

# include("thermal_diffusion/Rheology.jl")

include("IO/DataIO.jl")

end # module
