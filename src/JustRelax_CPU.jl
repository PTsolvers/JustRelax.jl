module JustRelax2D

using ..JustRelax
using JustPIC, JustPIC._2D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using Statistics
using MPI

import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
import JustRelax: PTStokesCoeffs
import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions

import JustPIC._2D:
    numphases,
    nphases

@init_parallel_stencil(Threads, Float64, 2)

include("common.jl")
include("stokes/Stokes2D.jl")
export solve!

end

module JustRelax3D

using ..JustRelax
using JustPIC, JustPIC._3D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using Statistics
using MPI

import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
import JustRelax: PTStokesCoeffs
import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions

import JustPIC._3D:
    numphases,
    nphases

@init_parallel_stencil(Threads, Float64, 3)

include("common.jl")
include("stokes/Stokes3D.jl")
export solve!

end
