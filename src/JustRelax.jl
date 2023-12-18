module JustRelax

using Reexport
@reexport using ParallelStencil
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using MPI
using GeoParams
using HDF5
using CellArrays
using StaticArrays
using JustPIC

function solve!() end

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry

include("MiniKernels.jl")
export _d_xa,
    _d_ya,
    _d_za,
    _d_za,
    _d_xi,
    _d_yi,
    _d_zi,
    _d_zi,
    _av,
    _av_a,
    _av_xa,
    _av_ya,
    _av_x,
    _av_y,
    _av_z,
    _av_yz,
    _av_xz,
    _av_xy,
    _av_yzi,
    _av_xzi,
    _av_xyi,
    _gather,
    _gather_yz,
    _gather_xz,
    _gather_xy,
    _harm_x,
    _harm_y,
    _harm_z,
    _harm_yz,
    _harm_xz,
    _harm_xy,
    _harm_xyi,
    _harm_xzi,
    _harm_yzi,
    _current

include("phases/CallArrays.jl")
export @cell, element, setelement!, cellnum, cellaxes, new_empty_cell, setindex!

include("rheology/StressUpdate.jl")
export plastic_params, plastic_params_phase, compute_dτ_r, _compute_τ_nonlinear!

include("MetaJustRelax.jl")

include("stokes/MetaStokes.jl")
export PS_Setup, environment!, ps_reset!

include("thermal_diffusion/MetaDiffusion.jl")

include("thermal_diffusion/Rheology.jl")

include("IO/DataIO.jl")

end # module
